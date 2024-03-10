import torch
import  time
import moviepy.editor as mp
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Dataset
import ffmpeg
import os
import datetime
import yaml
from pprint import pprint
from tqdm import tqdm

def process_file(folder, input_file, output_folder, soundtrack_folder, use_gpu=True, translate_to_french=True, debug=False):
    
    t0 = time.time()

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(soundtrack_folder):
        os.mkdir(soundtrack_folder)
    # filename = "YOUR_FILE"
    filename = input_file.split(".")[0]
    extension = input_file.split(".")[1]

    if use_gpu:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        device = "cpu"
        torch_dtype = torch.float32

    if not translate_to_french:
        output_srt_name = filename+"_EN.srt"
    else:
        output_srt_name = filename+".fre.srt"

    if not os.path.exists(os.path.join(soundtrack_folder, filename+".mp3")):
        print("Extracting soundtrack...", end="")
        my_clip = mp.VideoFileClip(os.path.join(folder, filename+"."+extension))
        my_clip.audio.write_audiofile(os.path.join(soundtrack_folder, filename+".mp3"))
        print("Done")

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Generating transcript...", end="")
    result = pipe(os.path.join(soundtrack_folder, filename+".mp3"), generate_kwargs={"language": "english"})
    print("Done")
    ## Pre-processing of the transcript: merging sentences that are too short.
    ## Pre-translation merging rules: Tries to make de printed sentences long enough to prevent flickering, and also improve translation (by having more context in a given chunk).
    minimum_timespan = 0.5 # In sec. How much time a single sentence should stay on screen. If too short, try to merge with the next one.
    maximum_timespan = 5 # In sec. Maximum time a merged segment can be. If an attempted merge ends up being more, revert.

    # Disabled for now
    #minimum_words = 10 # In words. How much words a printed sentence must contain at least. If too short, attempt merge with next one.
    #maximum_words = 9999999 # In words. How much words a printed sentence can contains at most. If an attempted merge ends up being more, revert.

    print("Post-processing transcript chunks...", end="")
    processed_result = {"chunks":[]}
    skip_x = 0
    for i in range(len(result["chunks"])-1):
        if skip_x > 0:
            skip_x -= 1
            continue
        chunk = result["chunks"][i]
        if None in chunk["timestamp"]:
            continue
        timespan = chunk["timestamp"][1] - chunk["timestamp"][0]
        nb_words = len(chunk["text"].split(" "))

        # j is to use to have a recursive merge if the result is still too short
        j = i
        if timespan < minimum_timespan:# or nb_words < minimum_words:    
            # Attempt merge with next sentence(s)
            next_chunk = result["chunks"][j+1]

            if None in next_chunk["timestamp"]:
                processed_result["chunks"].append(chunk)
                continue

            next_timespan = next_chunk["timestamp"][1] - next_chunk["timestamp"][0]
            next_nb_words = len(next_chunk["text"].split(" "))

            if timespan + next_timespan < maximum_timespan:
                # Sentence too fast to read, merge at any cost!
                if debug:
                    print("Merged sentences because timespan was too short")
                    print("Initial chunk:", chunk)
                chunk = {
                                "timestamp":(chunk["timestamp"][0], next_chunk["timestamp"][1]),
                                "text": chunk["text"] + next_chunk["text"]
                            }
                processed_result["chunks"].append(chunk)
                if debug:
                    print("Merged chunk:", chunk)
                skip_x = 1
                continue
            else:
                # Cancelling merge.
                processed_result["chunks"].append(chunk)

        else:
            # No merging to do
            processed_result["chunks"].append(chunk)
    print("Done")

    print("Translating...", end="")

    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-fr", batch_size=16)
    translation_timestamped = []

    for chunk in processed_result["chunks"]:
        if debug:
            print("\nOriginal:", chunk["text"])
        if translate_to_french:
            
            translation_timestamped.append([chunk["timestamp"],  pipe(chunk["text"])[0]["translation_text"]])
            
            # print(translation_timestamped[-1][1])
        else:
            translation_timestamped.append([chunk["timestamp"],  chunk["text"]])
        
        if debug:
            print("Timestamp:", chunk["timestamp"])
    print("Done")
    print("Writing srt...", end="")
    final = []
    i = 1
    for timestamp, text in translation_timestamped:
        if timestamp[0] is None or timestamp[1] is None:
            # print("Faulty timestamp:", timestamp)
            # print("text:", text)
            continue
        
        t1_str = str(datetime.timedelta(seconds=timestamp[0]))
        t2_str = str(datetime.timedelta(seconds=timestamp[1]))

        if "." in t1_str:
            t1_str = t1_str.split(".")[0] + "," + t1_str.split(".")[1][0:3]
        if "." in t2_str:
            t2_str = t2_str.split(".")[0] + "," + t2_str.split(".")[1][0:3]

        final.append(str(i))
        final.append(t1_str + " --> " + t2_str)
        final.append(text)
        final.append("")
        i+=1

    import codecs
    with codecs.open(os.path.join(output_folder, output_srt_name), 'w', 'utf-8') as the_file:
        for line in final:
            # print("line:", line)        
            the_file.write(line+'\n')
    print("Done")
    print("Elapsed time:", time.time()-t0)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config_file = "config.yaml"

    config = read_yaml(config_file)

    pprint(config)

    folder = "./"
    filename = "sample_10min.mkv"
    if config["FILES"]["process_single_file"]:
        process_file(
                        config["FILES"]["folder_path"], 
                        config["FILES"]["filename"], 
                        use_gpu=config["INFERENCE"]["use_gpu"],
                        translate_to_french=config["END_RESULT"]["translate_to_french"],
                        debug=config["INFERENCE"]["debug"])
    else:
        files = os.listdir(config["FILES"]["folder_path"])
        
        for file in tqdm(files):
            print("\n\nProcessing:", file)
            if config["FILES"]["skip_if_srt_exists"] and not os.path.exists(os.path.join(config["FILES"]["output_folder"], file.split(".")[0]+".fre.srt")):
                process_file(
                            config["FILES"]["folder_path"], 
                            file, 
                            config["FILES"]["output_folder"],
                            config["FILES"]["soundtrack_folder"],
                            use_gpu=config["INFERENCE"]["use_gpu"],
                            translate_to_french=config["END_RESULT"]["translate_to_french"],
                            debug=config["INFERENCE"]["debug"])
            else:
                print("SRT exists, skip.")
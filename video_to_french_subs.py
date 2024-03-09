# %%
import torch
torch.cuda.is_available()

import moviepy.editor as mp

#filename = "sample"
# filename = "sample_10min"
filename = "The Office S01E01 Downsize"
# translate_to_french = False
translate_to_french = True


if not translate_to_french:
    output_srt_name = filename+"_raw_EN.srt"
else:
    output_srt_name = filename+".srt"

# %%
import os 

if not os.path.exists(filename+".mp3"):
    my_clip = mp.VideoFileClip(filename+".mkv")
    my_clip.audio.write_audiofile(filename+".mp3")

# %%
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Dataset
import ffmpeg
import os

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu" # For testing!
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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

#dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
#sample = dataset[0]["audio"]
#result = pipe(sample)

#audio_dataset = Dataset.from_dict({"audio": ["sample.mp3"]})
#sample = audio_dataset[0]["audio"]

result = pipe(filename+".mp3")
# print(result)



# %%
## Pre-processing of the transcript: merging sentences that are too short.
## Pre-translation merging rules: Tries to make de printed sentences long enough to prevent flickering, and also improve translation (by having more context in a given chunk).
minimum_timespan = 0.5 # In sec. How much time a single sentence should stay on screen. If too short, try to merge with the next one.
maximum_timespan = 5 # In sec. Maximum time a merged segment can be. If an attempted merge ends up being more, revert.

# Disabled for now
#minimum_words = 10 # In words. How much words a printed sentence must contain at least. If too short, attempt merge with next one.
#maximum_words = 9999999 # In words. How much words a printed sentence can contains at most. If an attempted merge ends up being more, revert.

processed_result = {"chunks":[]}
skip_x = 0
for i in range(len(result["chunks"])):
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
            print("Merged sentences because timespan was too short")
            print("Initial chunk:", chunk)
            chunk = {
                            "timestamp":(chunk["timestamp"][0], next_chunk["timestamp"][1]),
                            "text": chunk["text"] + next_chunk["text"]
                        }
            processed_result["chunks"].append(chunk)
            print("Merged chunk:", chunk)
            skip_x = 1
            continue
        else:
            # Cancelling merge.
            processed_result["chunks"].append(chunk)
        
        ## Weird, this clause is not coherent with the previous one.
        # if nb_words < minimum_words:
        #     if len((chunk["text"] + next_chunk["text"]).split(" ")) < maximum_words:
        #         # Sentence too short
        #         chunk = {
        #                         "timestamp":(chunk["timestamp"][0], next_chunk["timestamp"][1]),
        #                         "text": chunk["text"] + next_chunk["text"]
        #                     }
        #         processed_result["chunks"].append(chunk)
        #         skip_x = 1
        #         continue
    else:
        # No merging to do
        processed_result["chunks"].append(chunk)
    
    # print(processed_result)


# %%
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-fr", batch_size=16)

translation_timestamped = []

for chunk in processed_result["chunks"]:
    print("\nOriginal:", chunk["text"])
    if translate_to_french:
        translation_timestamped.append([chunk["timestamp"],  pipe(chunk["text"])[0]["translation_text"]])
        print(translation_timestamped[-1][1])
    else:
        translation_timestamped.append([chunk["timestamp"],  chunk["text"]])
    print("Timestamp:", chunk["timestamp"])
    

# %%
import datetime
final = []
i = 1
for timestamp, text in translation_timestamped:
    if timestamp[0] is None or timestamp[1] is None:
        print("Faulty timestamp:", timestamp)
        print("text:", text)
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
with codecs.open(output_srt_name, 'w', 'utf-8') as the_file:
    for line in final:
        print("line:", line)        
        the_file.write(line+'\n')
        



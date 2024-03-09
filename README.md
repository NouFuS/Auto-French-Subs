# Automatic English Video to French SRT

Create French subtitles directly from a video file with English audio. (also possible to skip the translation to obtain the raw english subtitles)

## Quality: 
* English transcript: Good
* French translation: Moderate

## Performances
### Tested on the following hardware
* CPU: 12th Gen Intel(R) Core(TM) i9-12900HK   2.50 GHz
* GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU
* RAM: 64Go

### Test case:
10min video
* GPU: 1m15
* CPU: 5m50

# Workflow: 
* Extract the audio track
* Perform a transcription with OpenAI Whisper Large v3 from the HuggingFace Hub
* Post-process the timestamps to ensure readability of the subtitles
* Translate the transcript with Opus-mt-en-fr from the HuggingFace Hub
* Create the srt file

# Requirements
## Hardware
A modern GPU or 32Go of RAM for CPU (will be much much slower, at least 5x)

## Software environment
Tested on Ubuntu 22.04
### System-wide libraries
* ffmpeg
  * `sudo apt install ffmpeg`

### Python env
* torch with CUDA support (for GPU inference)
  * Follow the install documentation from [pytorch.org](https://pytorch.org/get-started/locally/)
* moviepy
  * `pip install moviepy`
* transformers library from HuggingFace
  *   `pip install transformers`
* datasets library from HuggingFace
  * `pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]`

# Usage
  Edit the video_to_french.py file to specify the file name.
  Run the file, it will create a srt file with the same name as the input video file.

 
This project was a testing ground to work with the HuggingFace plateform.

TODO: make a gradio GUI and a clean data input. For now, you need to modify variables the code itself.

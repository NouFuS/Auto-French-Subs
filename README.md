# Video to French SRT

Create English or French subtitles directly from a video file

# Workflow: 
* Extract the audio track
* Perform a transcription with OpenAI Whisper Large v3 from the HuggingFace Hub
* Post-process the timestamps to ensure readability of the subtitles
* Translate the transcript with Opus-mt-en-fr from the HuggingFace Hub
* Create the srt file

This project was a testing ground to work with the HuggingFace plateform.

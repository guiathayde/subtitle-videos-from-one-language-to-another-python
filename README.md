# Video Transcription and Translation

This project automates the process of:  
1. Transcribing videos to English subtitles using Whisper.  
2. Translating those English subtitles to Portuguese using MarianMT.  
3. Embedding the translated subtitles into the output video with FFmpeg.

## Tools Used

- [OpenAI Whisper](https://github.com/openai/whisper) for speech-to-text (audio → English text).  
- [Helsinki-NLP/MarianMT](https://huggingface.co/Helsinki-NLP) for machine translation (English → Portuguese).  
- [FFmpeg](https://ffmpeg.org/) for audio extraction and burning subtitles into the video.  
- [PyTorch](https://pytorch.org/) for running Whisper and MarianMT on GPU/CPU.  
- [Colorama](https://pypi.org/project/colorama/) for console colors.  

## Limitations

- Whisper might not accurately transcribe poor-quality audio or heavy accents.  
- Translation quality depends on the MarianMT model.  
- Large videos require sufficient processing power (especially if using GPU).  
- Segment splitting is limited to 10-second chunks, which may break longer sentences.

## How to Run

1. Clone or place this code in a local folder.  
2. Create a virtual environment and activate it:
   ```console
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```console
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```console
   python main.py
   ```
5. Enter the path to the directory containing .mp4 files and the destination directory for output.  

The script will:  
• Find .mp4 files, extract audio with FFmpeg, transcribe with Whisper, generate English .srt, translate to Portuguese .srt, and finally embed subtitles into a new .mp4 file.  
• Skip each step if the corresponding output file already exists. 

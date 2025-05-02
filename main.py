import os
import whisper        # Whisper for speech-to-text
import subprocess     # To run external commands (like ffmpeg)
import torch          # PyTorch library
import datetime       # Work with date/time objects
from colorama import Fore, Style, init
from transformers import MarianMTModel, MarianTokenizer
import concurrent.futures
import hashlib
import shlex
import math
import platform

# Initialize colorama (ANSI color codes)
init(autoreset=True)

# Logging functions for console output
def log_info(msg): print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {msg}")
def log_success(msg): print(f"{Fore.GREEN}[OK]{Style.RESET_ALL} {msg}")
def log_warning(msg): print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {msg}")
def log_error(msg): print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")

def load_translate_model():
    """
    Loads the Marian translation model (English->Portuguese).
    Moves it to CUDA if available, or uses CPU otherwise.
    """
    model_name = 'Helsinki-NLP/opus-mt-tc-big-en-pt'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

def translate_phrase(phrase, model, tokenizer, cache):
    """
    Translates a single phrase using MarianMT.
    Uses an MD5-based cache to avoid repeated translation of identical text.
    """
    phrase = phrase.strip()
    if not phrase:
        return ""

    key = hashlib.md5(phrase.encode('utf-8')).hexdigest()
    if key in cache:
        return cache[key]

    # Tokenize and generate translation
    tokens = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    translated = model.generate(**tokens)
    text = tokenizer.decode(translated[0], skip_special_tokens=True)
    cache[key] = text
    return text

def translate_srt(input_path, output_path):
    """
    Reads an .srt file, splits it into text blocks, 
    translates them with MarianMT, then writes the translated .srt.
    """
    model, tokenizer = load_translate_model()
    cache = {}

    # Read all lines from the input SRT
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Separate lines indicating times / numbers from text
    blocks = []
    current_block = []
    lines_out = []
    for line in lines:
        if line.strip().isdigit() or "-->" in line:
            if current_block:
                blocks.append(" ".join(current_block).strip())
                current_block = []
            lines_out.append(line)
        elif line.strip() == "":
            if current_block:
                blocks.append(" ".join(current_block).strip())
                current_block = []
            lines_out.append(line)
        else:
            current_block.append(line.strip())
    if current_block:
        blocks.append(" ".join(current_block).strip())

    # Translate blocks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        translations = list(executor.map(lambda txt: translate_phrase(txt, model, tokenizer, cache), blocks))

    # Write out lines, replacing original text with translations
    idx_block = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in lines:
            if line.strip().isdigit() or "-->" in line or line.strip() == "":
                outfile.write(line)
            else:
                if idx_block < len(translations):
                    outfile.write(translations[idx_block] + '\n')
                    idx_block += 1
    log_success(f"Local translation completed for {input_path}. Stored in {output_path}.")

def list_mp4_videos(directory):
    """
    Recursively lists all .mp4 files in the specified directory.
    """
    files_mp4 = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp4"):
                full_path = os.path.join(root, file)
                files_mp4.append(full_path)
    return files_mp4

def extract_audio(video_path, audio_path):
    """
    Uses ffmpeg to extract audio from an .mp4 and save it as WAV 16kHz.
    Skips extraction if audio file already exists.
    """
    if os.path.exists(audio_path):
        log_info(f"Audio file {audio_path} already exists. Skipping extraction.")
        return audio_path

    log_info(f"Extracting audio from {video_path} to {audio_path}...")
    comando = [
        "ffmpeg", "-hwaccel", "cuda", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    result = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        log_error(f"Error executing ffmpeg:\n{result.stderr.decode()}")
        raise RuntimeError("ffmpeg failed to extract audio.")
    
    return audio_path

def format_timestamp(seconds):
    """
    Converts a float 'seconds' into SRT-friendly time format, e.g. 00:00:01,234
    """
    td = datetime.timedelta(seconds=int(seconds))
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{str(td)}.{milliseconds:03}".replace(".", ",")

def save_srt_old(segments, output_path):
    """
    Writes SRT file without splitting segments exceeding 10 seconds.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def save_srt(segments, output_path):
    """
    Writes SRT file, splitting segments exceeding 10 seconds 
    into multiple blocks. Also breaks text proportionally per block.
    """
    max_duration = 10.0  # Max 10 seconds per block
    with open(output_path, "w", encoding="utf-8") as f:
        srt_index = 1
        for segment in segments:
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            duration = end - start

            # If duration <= 10s, write as is
            if duration <= max_duration:
                start_str = format_timestamp(start)
                end_str = format_timestamp(end)
                f.write(f"{srt_index}\n{start_str} --> {end_str}\n{text}\n\n")
                srt_index += 1
            else:
                # Split into multiple 10s blocks
                num_blocks = math.ceil(duration / max_duration)
                words = text.split()
                words_per_block = math.ceil(len(words) / num_blocks)

                for i in range(num_blocks):
                    sub_start = start + (i * max_duration)
                    sub_end = min(sub_start + max_duration, end)
                    start_str = format_timestamp(sub_start)
                    end_str = format_timestamp(sub_end)

                    # Grab only part of the words for each sub-block
                    start_words = i * words_per_block
                    end_words = min((i + 1) * words_per_block, len(words))
                    sub_text = " ".join(words[start_words:end_words])

                    f.write(f"{srt_index}\n{start_str} --> {end_str}\n{sub_text}\n\n")
                    srt_index += 1

def transcribe_video_to_srt(video_path, language, output_audio, output_srt):
    """
    Transcribes the input video using Whisper, 
    saves transcript as SRT. Skips if the output SRT already exists.
    """
    if os.path.exists(output_srt):
        log_info(f"Transcription file {output_srt} already exists. Skipping transcription.")
        return

    log_info("Extracting audio...")
    audio_path = extract_audio(video_path, audio_path=output_audio)

    log_info("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium',
    # 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'
    model = whisper.load_model("tiny.en", device=device)
    log_info(f"Transcribing with Whisper ({device})...")
    result = model.transcribe(audio_path, language=language, verbose=False)

    log_info(f"Saving transcript to {output_srt}")
    save_srt(result["segments"], output_srt)
    log_success("Transcription completed.")

def embed_caption(video_path, srt_path, output_path):
    """
    Uses ffmpeg to burn subtitles from .srt into the video.
    """
    srt_path_ffmpeg = srt_path.replace(os.sep, '/').replace(':', '\\:')
    vf_filter = f"subtitles={shlex.quote(srt_path_ffmpeg)}:charenc=UTF-8"

    system = platform.system().lower()
    if system == "windows":
        comando = [
            "ffmpeg", "-hwaccel", "cuda", "-i", video_path,
            "-vf", vf_filter,
            "-c:v", "h264_nvenc", "-preset", "fast", "-b:v", "5M",
            "-c:a", "copy", output_path
        ]
    else:
        # For macOS and Linux, use libx264 (CPU)
        comando = [
            "ffmpeg", "-i", video_path,
            "-vf", vf_filter,
            "-c:v", "libx264", "-preset", "fast", "-b:v", "5M",
            "-c:a", "copy", output_path
        ]

    log_info(f"Embedding subtitles in {output_path}...")
    subprocess.run(comando)
    log_success("Embedded subtitle successful!")

# === RUNNING SECTION ===
# Prompt user for directories
movies_directory = input("Enter the path to the movies directory: ")
translated_movies_directory = input("Enter the path to the translated movies directory: ")

# Check if the original directory exists
if not os.path.exists(movies_directory):
    log_error(f"Directory {movies_directory} does not exist.")
    exit(1)

# Prepare output directories
audio_path = os.path.join(translated_movies_directory, "audio")
en_srt_path = os.path.join(translated_movies_directory, "en_srt")
pt_srt_path = os.path.join(translated_movies_directory, "pt_srt")
video_path = os.path.join(translated_movies_directory, "video")

os.makedirs(translated_movies_directory, exist_ok=True)
os.makedirs(audio_path, exist_ok=True)
os.makedirs(en_srt_path, exist_ok=True)
os.makedirs(pt_srt_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# Gather all .mp4 files
videos_paths = list_mp4_videos(movies_directory)
log_info(f"{len(videos_paths)} video(s) found.")

# Process each video
for path in videos_paths:
    try:
        nome_base = os.path.splitext(os.path.basename(path))[0]
        output_audio = os.path.join(audio_path, f"{nome_base}_audio.wav")
        output_en_srt = os.path.join(en_srt_path, f"{nome_base}_en.srt")
        output_pt_srt = os.path.join(pt_srt_path, f"{nome_base}_pt.srt")
        output_video = os.path.join(video_path, f"{nome_base}_legendado.mp4")

        log_info(f"\n📁 Processing: {nome_base}")
        
        # Check if final .mp4 already exists
        if os.path.exists(output_video):
            log_info(f"File {output_video} already exists. Skipping processing.")
            continue

        # Check if Portuguese subtitle already exists
        if os.path.exists(output_pt_srt):
            log_info(f"File {output_pt_srt} already exists. Skipping translation.")
            embed_caption(path, output_pt_srt, output_video)
            continue

        # Check if English subtitle file already exists
        if os.path.exists(output_en_srt):
            log_info(f"File {output_en_srt} already exists. Skipping transcription.")
            translate_srt(output_en_srt, output_pt_srt)
            embed_caption(path, output_pt_srt, output_video)
            continue

        # If none of them exists, proceed with transcription, translation, and embedding
        transcribe_video_to_srt(path, language="en", output_audio=output_audio, output_srt=output_en_srt)
        translate_srt(output_en_srt, output_pt_srt)
        embed_caption(path, output_pt_srt, output_video)

        log_success(f"✅ Finished: {nome_base}\n")

    except Exception as e:
        log_error(f"❌ Failed to process {path}: {e}")
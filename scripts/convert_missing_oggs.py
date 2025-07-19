import os
import subprocess

FFMPEG_PATH = r"D:\taiko_ai\taiko-autochart\tools\ffmpeg\bin\ffmpeg.exe"  # <-- Your FFmpeg path
SAMPLE_RATE = 22050

def convert_ogg_to_wav(ogg_path):
    wav_path = os.path.splitext(ogg_path)[0] + ".wav"
    if os.path.exists(wav_path):
        print(f"✅ Already exists: {wav_path}")
        return False

    if not os.path.exists(ogg_path):
        print(f"❌ File not found: {ogg_path}")
        return False

    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    command = [
        FFMPEG_PATH,
        "-y",
        "-i", ogg_path,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        wav_path
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        print(f"🎧 Converted: {ogg_path} -> {wav_path}")
        return True
    else:
        print(f"❌ Conversion failed: {ogg_path}")
        return False

def process_missing_file_list(log_path="missing_files.log"):
    if not os.path.exists(log_path):
        print("🚫 Log file not found!")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    converted = 0
    skipped = 0
    failed = 0

    for ogg_path in lines:
        if convert_ogg_to_wav(ogg_path):
            converted += 1
        else:
            if os.path.exists(os.path.splitext(ogg_path)[0] + ".wav"):
                skipped += 1
            else:
                failed += 1

    print("\n✅ Done!")
    print(f"🎧 Converted: {converted}")
    print(f"🔁 Skipped (already exists): {skipped}")
    print(f"❌ Failed or missing: {failed}")

if __name__ == "__main__":
    process_missing_file_list()

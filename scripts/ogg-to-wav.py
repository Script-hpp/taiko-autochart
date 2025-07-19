import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

FFMPEG_PATH = r"D:\taiko_ai\taiko-autochart\tools\ffmpeg\bin\ffmpeg.exe"  # Change this!

def convert_file(src_path, sample_rate=22050):
    dst_path = os.path.splitext(src_path)[0] + ".wav"  # same folder, same name but .wav
    command = [
        FFMPEG_PATH,
        "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        dst_path
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        print(f"✅ Converted: {src_path} -> {dst_path}")
    else:
        print(f"❌ Failed: {src_path}")

def find_ogg_files(src_dir):
    ogg_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(".ogg"):
                ogg_files.append(os.path.join(root, file))
    return ogg_files

def convert_ogg_to_wav_parallel(src_dir, sample_rate=22050, max_workers=6, limit=None):
    ogg_files = find_ogg_files(src_dir)
    if limit:
        ogg_files = ogg_files[:limit]
    print(f"Found {len(ogg_files)} .ogg files to convert.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_file, src_path, sample_rate) for src_path in ogg_files]
        for future in as_completed(futures):
            pass

    print("✅ All conversions complete.")

if __name__ == "__main__":
    convert_ogg_to_wav_parallel("dataset-dirty", limit=2740)

import os
import re
import shutil
import subprocess
import difflib

FFMPEG_PATH = r"D:\taiko_ai\taiko-autochart\tools\ffmpeg\bin\ffmpeg.exe"  # Set your FFmpeg path
SAMPLE_RATE = 22050

def extract_ogg_names_from_tja(tja_path):
    ogg_files = set()
    with open(tja_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            wave_match = re.search(r'WAVE\s*:\s*(.+\.ogg)', line, re.IGNORECASE)
            if wave_match:
                ogg_files.add(wave_match.group(1).strip())
            nextsong_match = re.search(r'#NEXTSONG[^,]*,[^,]*,[^,]*,\s*(.+\.ogg)', line, re.IGNORECASE)
            if nextsong_match:
                ogg_files.add(nextsong_match.group(1).strip())
    return ogg_files

def fuzzy_match_file(filename, file_list, cutoff=0.6):
    """Find the closest match for a file in the list, ignoring exact case/spacing issues."""
    matches = difflib.get_close_matches(filename, file_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def convert_ogg_to_wav(ogg_path):
    wav_path = os.path.splitext(ogg_path)[0] + ".wav"
    if os.path.exists(wav_path):
        return True

    if not os.path.exists(ogg_path):
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
    return result.returncode == 0 and os.path.exists(wav_path)

def process_folder(folder_path, missing_log):
    tja_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tja")]
    if not tja_files:
        return False

    ogg_needed = set()
    for tja in tja_files:
        tja_path = os.path.join(folder_path, tja)
        ogg_needed.update(extract_ogg_names_from_tja(tja_path))

    all_files = os.listdir(folder_path)
    all_ok = True

    for ogg_expected in ogg_needed:
        match = fuzzy_match_file(ogg_expected, all_files)

        if not match:
            print(f"❌ No match for .ogg: {ogg_expected}")
            missing_log.append(os.path.join(folder_path, ogg_expected))
            all_ok = False
            continue

        ogg_path = os.path.join(folder_path, match)
        success = convert_ogg_to_wav(ogg_path)
        if success:
            print(f"🎧 Converted (fuzzy matched): {match}")
        elif os.path.exists(os.path.splitext(ogg_path)[0] + ".wav"):
            print(f"✅ Already exists: {match}")
        else:
            print(f"❌ Failed to convert: {match}")
            all_ok = False

    return all_ok

def copy_folder(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        shutil.copytree(src_folder, dst_folder)
        print(f"📦 Copied: {os.path.basename(src_folder)}")

def build_dataset_with_fuzzy_fix(src_dir, dst_dir, limit=None):
    os.makedirs(dst_dir, exist_ok=True)
    missing_log = []

    subfolders = [
        os.path.join(src_dir, name)
        for name in sorted(os.listdir(src_dir))
        if os.path.isdir(os.path.join(src_dir, name))
    ]

    copied = 0
    for folder in subfolders:
        if limit and copied >= limit:
            break
        if process_folder(folder, missing_log):
            rel_path = os.path.relpath(folder, src_dir)
            dst_path = os.path.join(dst_dir, rel_path)
            copy_folder(folder, dst_path)
            copied += 1

    print(f"\n✅ Copied {copied} folders to {dst_dir}")

    if missing_log:
        with open("missing_files.log", "w", encoding="utf-8") as f:
            for path in missing_log:
                f.write(path + "\n")
        print(f"⚠️  Still missing some files. Logged to missing_files.log")

if __name__ == "__main__":
    SOURCE_DIR = "dataset-dirty"
    DEST_DIR = "dataset-semi"
    LIMIT = None  # Change or set to None for full run

    build_dataset_with_fuzzy_fix(SOURCE_DIR, DEST_DIR, limit=LIMIT)

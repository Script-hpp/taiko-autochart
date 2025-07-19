import os
from pydub import AudioSegment
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def find_files_with_ext(root_dir, ext):
    result = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(ext.lower()):
                result.append(os.path.join(root, f))
    return result

def check_wav_for_ogg(ogg_path):
    wav_path = os.path.splitext(ogg_path)[0] + ".wav"
    if not os.path.isfile(wav_path):
        return False, None
    try:
        AudioSegment.from_wav(wav_path)
        return True, wav_path
    except:
        return False, wav_path

def compare_durations(ogg_path, wav_path, tolerance_ms=50):
    try:
        ogg_audio = AudioSegment.from_ogg(ogg_path)
        wav_audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        return False, 0, 0
    diff = abs(len(ogg_audio) - len(wav_audio))
    return diff <= tolerance_ms, len(ogg_audio), len(wav_audio)

def check_tja_for_folder(folder):
    for f in os.listdir(folder):
        if f.lower().endswith(".tja"):
            return True
    return False

def fuzzy_find_audio_for_tja(tja_audio_name, audio_file_list, threshold=80):
    # Use fuzzy matching to find closest audio filename match in list
    matches = process.extract(tja_audio_name, audio_file_list, scorer=fuzz.ratio)
    for match_name, score in matches:
        if score >= threshold:
            return match_name
    return None

def parse_tja_for_audio_references(tja_path):
    # Parse lines that start with "WAVE:" or #NEXTSONG line that may contain audio filename
    audio_files = set()
    try:
        with open(tja_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("WAVE:"):
                    wave_file = line[5:].strip()
                    if wave_file:
                        audio_files.add(wave_file)
                elif line.startswith("#NEXTSONG"):
                    parts = line.split(",")
                    if len(parts) >= 4:
                        wave_file = parts[3].strip()
                        if wave_file.lower().endswith(('.ogg', '.wav')):
                            audio_files.add(wave_file)
    except Exception as e:
        print(f"Error reading {tja_path}: {e}")
    return audio_files

def main(dataset_root):
    print(f"Scanning dataset root: {dataset_root}")

    ogg_files = find_files_with_ext(dataset_root, ".ogg")
    wav_files = find_files_with_ext(dataset_root, ".wav")
    tja_files = find_files_with_ext(dataset_root, ".tja")

    print(f"Found {len(ogg_files)} .ogg files")
    print(f"Found {len(wav_files)} .wav files")
    print(f"Found {len(tja_files)} .tja files")

    wav_files_lower = [os.path.basename(f).lower() for f in wav_files]

    missing_wav = []
    corrupted_wav = []
    duration_mismatches = []
    missing_tja = []
    fuzzy_missing_audio = []

    # Check .wav files for every .ogg
    for ogg_path in ogg_files:
        has_wav, wav_path = check_wav_for_ogg(ogg_path)
        if not has_wav:
            missing_wav.append(os.path.splitext(ogg_path)[0] + ".wav")
            continue
        # check if corrupted
        if wav_path is None:
            corrupted_wav.append(wav_path)
            continue
        # check durations
        ok, ogg_len, wav_len = compare_durations(ogg_path, wav_path)
        if not ok:
            duration_mismatches.append((ogg_path, wav_path, ogg_len, wav_len))

    # Check for .tja files presence per folder
    # Assuming each song folder contains audio + .tja
    song_folders = set(os.path.dirname(f) for f in ogg_files)
    for folder in song_folders:
        if not check_tja_for_folder(folder):
            missing_tja.append(folder)

    # Fuzzy match audio referenced in tja files
    all_audio_files = [os.path.basename(f) for f in ogg_files + wav_files]
    for tja_path in tja_files:
        audio_refs = parse_tja_for_audio_references(tja_path)
        for ref in audio_refs:
            # Check if exact match exists
            if ref not in all_audio_files:
                # Try fuzzy match
                match = fuzzy_find_audio_for_tja(ref, all_audio_files)
                if not match:
                    fuzzy_missing_audio.append((tja_path, ref))

    # Reporting
    print("\n=== REPORT ===")
    print(f"Missing .wav files: {len(missing_wav)}")
    for f in missing_wav:
        print(f"  ❌ {f}")

    print(f"\nCorrupted .wav files: {len(corrupted_wav)}")
    for f in corrupted_wav:
        print(f"  ❌ {f}")

    print(f"\nDuration mismatches (>50ms): {len(duration_mismatches)}")
    for ogg_p, wav_p, ogg_len, wav_len in duration_mismatches:
        print(f"  ⚠️ {ogg_p} vs {wav_p} | ogg={ogg_len}ms wav={wav_len}ms")

    print(f"\nFolders missing .tja files: {len(missing_tja)}")
    for folder in missing_tja:
        print(f"  ❌ {folder}")

    print(f"\nAudio referenced in .tja but missing or no close match: {len(fuzzy_missing_audio)}")
    for tja_file, audio_ref in fuzzy_missing_audio:
        print(f"  ❌ {tja_file} references missing audio file '{audio_ref}'")

    print("\nDone.")

if __name__ == "__main__":
    DATASET_PATH = "dataset-semi"  # Change this to your dataset folder path
    main(DATASET_PATH)

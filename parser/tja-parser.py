import os
import re
import torch
from pathlib import Path

VALID_NOTES = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}  # all supported symbols
SUPPORTED_COURSES = {"Easy", "Normal", "Hard", "Oni", "Ura", "Edit"}

def parse_tja_file(tja_path):
    charts = []
    current_course = None
    collecting = False
    current_chart = []

    with open(tja_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('//'):
                continue

            # Detect course header
            if line.startswith('COURSE:'):
                course_name = line.split(':')[1].strip()
                if course_name not in SUPPORTED_COURSES:
                    current_course = None
                else:
                    current_course = course_name
                continue

            if '#START' in line and current_course:
                collecting = True
                current_chart = []
                continue

            if '#END' in line and collecting:
                collecting = False
                if current_chart:
                    charts.append((current_course, current_chart))
                current_chart = []
                continue

            # Handle chart lines (e.g. 10101010,)
            if collecting:
                line = line.split('//')[0]  # remove inline comments
                for section in line.split(','):
                    section = section.strip()
                    if section and all(c in VALID_NOTES for c in section):
                        current_chart.append([int(c) for c in section])

    return charts

def save_label_tensor(chart_lines, output_path):
    # Pad lines to max length for consistent tensor shape
    if not chart_lines:
        return False

    max_len = max(len(line) for line in chart_lines)
    padded = [line + [0] * (max_len - len(line)) for line in chart_lines]
    tensor = torch.tensor(padded, dtype=torch.long)

    os.makedirs(output_path.parent, exist_ok=True)
    torch.save(tensor, output_path)
    return True

def process_dataset_tja(dataset_dir, output_root):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".tja"):
                tja_path = Path(root) / file
                relative = tja_path.relative_to(dataset_dir)
                song_name = tja_path.stem
                output_path = Path(output_root) / relative.with_suffix(".pt")

                print(f"Parsing: {tja_path}")
                try:
                    charts = parse_tja_file(tja_path)
                    if not charts:
                        print(f"⚠️  Skipping {tja_path} (no chart lines found)")
                        continue

                    # Save only the first course (you can change this logic)
                    course_name, chart_lines = charts[0]
                    success = save_label_tensor(chart_lines, output_path)
                    if success:
                        print(f"✅ Saved labels tensor: {output_path}")
                    else:
                        print(f"⚠️  Skipped (empty chart): {tja_path}")

                except Exception as e:
                    print(f"❌ Error parsing {tja_path}: {e}")

# === Run Script ===
if __name__ == "__main__":
    dataset_dir = "dataset-semi"             # <- Change to your dataset path
    output_dir = "dataset-labels-pt"         # <- Change output if needed
    process_dataset_tja(dataset_dir, output_dir)

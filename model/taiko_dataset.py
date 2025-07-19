import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Example dataset class (adjust paths and loading as needed)
class TaikoDataset(Dataset):
    def __init__(self, audio_root, label_root):
        self.audio_root = audio_root
        self.label_root = label_root

        # List all mel feature files (.pt)
        self.audio_files = []
        for root, _, files in os.walk(audio_root):
            for file in files:
                if file.endswith(".pt"):
                    self.audio_files.append(os.path.join(root, file))

        # Prepare corresponding label files assuming similar names + subdirs
        self.label_files = []
        for audio_path in self.audio_files:
            # Extract filename and assume label file matches relative path + filename
            rel_path = os.path.relpath(audio_path, audio_root)
            label_path = os.path.join(label_root, rel_path)
            if os.path.exists(label_path):
                self.label_files.append(label_path)
            else:
                # If label missing, skip this sample (or handle differently)
                self.label_files.append(None)

        # Filter out pairs where label file is None
        filtered_pairs = [(a, l) for a, l in zip(self.audio_files, self.label_files) if l is not None]
        self.audio_files, self.label_files = zip(*filtered_pairs)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label_path = self.label_files[idx]

        audio = torch.load(audio_path)  # Expected shape: [1, n_mels, time]
        label = torch.load(label_path)  # Adjust according to your label tensor shape

        return audio, label


def pad_collate(batch):
    audios = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad audios on the time dimension (last dim)
    max_audio_len = max(a.shape[-1] for a in audios)
    padded_audios = []
    for a in audios:
        pad_len = max_audio_len - a.shape[-1]
        if pad_len > 0:
            # pad last dim: (left_pad, right_pad)
            padded = F.pad(a, (0, pad_len))
        else:
            padded = a
        padded_audios.append(padded)
    batch_audios = torch.stack(padded_audios)

    # Pad labels on BOTH dimensions: sequence length (dim 0) AND features (dim 1)
    max_label_seq_len = max(l.shape[0] for l in labels)
    max_label_features = max(l.shape[1] for l in labels)
    
    padded_labels = []
    for l in labels:
        seq_pad_len = max_label_seq_len - l.shape[0]
        feat_pad_len = max_label_features - l.shape[1]
        
        # F.pad format for 2D tensor: (pad_left, pad_right, pad_top, pad_bottom)
        # We want to pad: right side of features (dim 1) and bottom of sequences (dim 0)
        padded = F.pad(l, (0, feat_pad_len, 0, seq_pad_len))
        padded_labels.append(padded)
    
    batch_labels = torch.stack(padded_labels)

    return batch_audios, batch_labels


if __name__ == "__main__":
    audio_root = r"D:\taiko_ai\taiko-autochart\mel_features"
    label_root = r"D:\taiko_ai\taiko-autochart\dataset-labels-pt"

    dataset = TaikoDataset(audio_root=audio_root, label_root=label_root)

    print(f"📦 Loaded {len(dataset)} audio-label pairs")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)

    # Iterate over the DataLoader
    for audio_batch, label_batch in loader:
        print("🎧 audio batch shape:", audio_batch.shape)
        print("🎯 label batch shape:", label_batch.shape)
        break
import os
import torch
import librosa
import torchaudio.transforms as T

def extract_mel_spectrogram(wav_path, sample_rate=22050, n_mels=128):
    waveform, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    waveform = torch.tensor(waveform).unsqueeze(0)
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512
    )
    mel = mel_spectrogram(waveform)
    return mel

def process_dataset_wavs(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_path = os.path.join(root, file)
                print(f"Processing {wav_path} ...")
                mel = extract_mel_spectrogram(wav_path)
                
                # Match label structure
                rel_path = os.path.relpath(wav_path, dataset_dir)
                rel_dir = os.path.dirname(rel_path)

                filename_no_ext = os.path.splitext(file)[0]  # removes .wav
                output_subdir = os.path.join(output_dir, rel_dir)
                os.makedirs(output_subdir, exist_ok=True)

                output_path = os.path.join(output_subdir, filename_no_ext + ".pt")
                torch.save(mel, output_path)

if __name__ == "__main__":
    dataset_dir = "dataset-semi"
    output_dir = "mel_features"
    process_dataset_wavs(dataset_dir, output_dir)

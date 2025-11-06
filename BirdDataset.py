import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch.nn.functional as F

class BirdDataset(Dataset):
    def __init__(
        self,
        data_path,
        annotations_file,
        split="train",
        sr=32000,
        window_len=5.0,
        hop_length=256,
        n_mels=128,
        transform=None,
        label_to_idx=None,
        encode_labels=False,
        pad=None,                          # None | "pad" | "squeeze" | "auto"
        precomputed=False,
        precomputed_dir="precomputed_mels",
    ):
        """
        annotations_file: CSV with columns [Filename, Start Time (s), End Time (s), Species eBird Code, where]
        precomputed: if True, loads precomputed mel .pt files saved as {row_id:06d}.pt
        pad: None (no change), 'pad' (pad short), 'squeeze' (truncate long), 'auto' (both)
        """
        df_full = pd.read_csv(annotations_file)
        # create stable row_id BEFORE filtering so it matches precompute filenames
        df_full = df_full.reset_index().rename(columns={"index": "row_id"})
        self.df = df_full[df_full["where"] == split].reset_index(drop=True)

        self.data_path = data_path
        self.sr = sr
        self.window_len = window_len
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.encode_labels = encode_labels
        self.pad = pad
        self.precomputed = precomputed
        self.precomputed_dir = precomputed_dir

        if not self.precomputed:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_fft=1024, hop_length=self.hop_length, n_mels=self.n_mels
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_id = int(row["row_id"])
        label_str = row["Species eBird Code"]
        label = (
            self.label_to_idx[label_str]
            if self.encode_labels and self.label_to_idx is not None
            else label_str
        )

        if self.precomputed:
            mel_path = os.path.join(self.precomputed_dir, f"{row_id:06d}.pt")
            if not os.path.exists(mel_path):
                raise FileNotFoundError(f"Missing precomputed mel: {mel_path}")
            mel = torch.load(mel_path, map_location="cpu")
        else:
            filename = os.path.join(self.data_path, row["Filename"])
            start, end = float(row["Start Time (s)"]), float(row["End Time (s)"])

            waveform, file_sr = torchaudio.load(filename)
            if file_sr != self.sr:
                waveform = torchaudio.transforms.Resample(file_sr, self.sr)(waveform)

            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)
            clip = waveform[:, start_sample:end_sample]

            mel = self.db_transform(self.mel_transform(clip))

        # padding
        if self.pad is not None:
            max_frames = int(self.window_len * self.sr / self.hop_length)
            T = mel.size(-1)
            if self.pad == "auto":
                if T < max_frames:
                    mel = F.pad(mel, (0, max_frames - T))
                elif T > max_frames:
                    mel = mel[..., :max_frames]
            elif self.pad == "pad" and T < max_frames:
                mel = F.pad(mel, (0, max_frames - T))
            elif self.pad == "squeeze" and T > max_frames:
                mel = mel[..., :max_frames]

        if self.transform is not None:
            mel = self.transform(mel)

        return mel, label
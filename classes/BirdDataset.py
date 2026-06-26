import os
import json
import random
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import torch.nn.functional as F

from common.common_utils import _mel_bin_centers_hz

class BirdDataset(Dataset):
    """
    Modes:
      (A) precomputed random windows: load 'precomputed_dir/manifest.csv' + samples/*.pt
          - returns:
              target_type='tf'  -> (mel: (1,F,T), target: (C,F,T))
              target_type='time'-> (mel: (1,F,T), target: (C,T))  [F-reduced]
      (B) on-the-fly from annotations_file (single-file windows at row start):
          - same return shapes as above
    """
    def __init__(
        self,
        data_path,
        annotations_file=None,
        split="train",
        sr=32000,
        hop_length=256,
        n_mels=128,
        n_fft=1024,
        window_len=10.0,
        label_to_idx=None,
        target_type="tf",
        include_overlaps=True,
        precomputed=False,
        precomputed_dir=None,
        fmin=0.0,
        fmax=None,
    ):

        # ---- initsss ----
        self.data_path = data_path
        self.sr, self.hop, self.n_mels, self.n_fft = sr, hop_length, n_mels, n_fft
        self.window_len = float(window_len)
        assert label_to_idx is not None, "label_to_idx required"
        self.label_to_idx = label_to_idx
        self.target_type = target_type
        self.include_overlaps = include_overlaps
        self.precomputed = precomputed
        self.precomputed_dir = precomputed_dir

        self.max_frames = int(round(self.window_len * self.sr / self.hop))
        self.C = len(self.label_to_idx)
        self.mel_centers_hz = _mel_bin_centers_hz(self.n_mels, self.sr, fmin, fmax)  # (F,)

        if ".csv" not in annotations_file :
            annotations_file = annotations_file + ".csv"

        self.use_manifest = False
        if self.precomputed:
            if self.precomputed_dir is None:
                raise ValueError("precomputed=True but precomputed_dir is None")
            man_path = os.path.join(self.precomputed_dir, "manifest.csv")
            if os.path.isfile(man_path):
                # ---- manifest mode (random windows) ----
                if annotations_file is None:
                    raise ValueError("annotations_file required to infer split from row_ids_json")
                man = pd.read_csv(man_path).reset_index(drop=True)

                # if split missing in manifest, infer from annotations via row_ids_json
                if "split" not in man.columns:
                    ann = pd.read_csv(annotations_file).reset_index().rename(columns={"index": "row_id"})
                    if "where" not in ann.columns:
                        ann["where"] = "train"
                    where_by_id = dict(zip(ann["row_id"].astype(int), ann["where"].astype(str)))

                    def _parse_ids(s):
                        try:
                            v = json.loads(s)
                            return [int(x) for x in v] if isinstance(v, list) else []
                        except Exception:
                            return []

                    def _infer_split(row):
                        ids = _parse_ids(row.get("row_ids_json", "[]"))
                        labs = [where_by_id.get(i) for i in ids if i in where_by_id]
                        labs = [x for x in labs if isinstance(x, str)]
                        if labs and all(l == "train" for l in labs): return "train"
                        if labs and all(l == "val" for l in labs): return "val"
                        
                        # fallback random
                        return "train" if random.randint(0, 1) else "val"

                    man["split"] = man.apply(_infer_split, axis=1)

                # filter by requested split
                man = man[man["split"] == split].reset_index(drop=True)
                self.manifest = man
                self.manifest["sample_path"] = self.manifest["sample_id"].apply(
                    lambda sid: os.path.join(self.precomputed_dir, "samples", f"{sid}.pt")
                )
                self.use_manifest = True
        else:
            # ---- on-the-fly from annotations ----
            if annotations_file is None:
                raise ValueError("annotations_file required for on-the-fly mode")
            df = pd.read_csv(annotations_file).reset_index().rename(columns={"index": "row_id"})
            self.df_full = df
            self.df = df[df["where"] == split].reset_index(drop=True)
            self._by_file = {k: v for k, v in df.groupby("Filename")}
            self.mel_tf = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mels, center=False
            )
            self.db_tf = torchaudio.transforms.AmplitudeToDB()


    def __len__(self):
        if self.precomputed and self.use_manifest:
            return len(self.manifest)
        return len(self.df)

    def _paint_tf(self, target, boxes, win_s, win_e, T):
        centers = self.mel_centers_hz  # (F,)
        for s, e, f_lo, f_hi, sp in boxes:
            c = self.label_to_idx.get(sp, None)
            if c is None: 
                continue
            # map time
            s_clip, e_clip = max(s, win_s), min(e, win_e)
            if e_clip <= s_clip:
                continue
            sf = int(((s_clip - win_s) * self.sr) // self.hop)
            ef = int(np.ceil(((e_clip - win_s) * self.sr) / self.hop))
            sf = max(0, min(T, sf)); ef = max(0, min(T, ef))
            if ef <= sf:
                continue
            # map freq via bin centers
            fmask = (centers >= f_lo) & (centers <= f_hi)
            if fmask.any():
                target[c, fmask, sf:ef] = 1
        return target

    def __getitem__(self, idx):
        # --------- (A) precomputed random windows via manifest -----------
        if self.precomputed and self.use_manifest:
            blob = torch.load(self.manifest.loc[idx, "sample_path"], map_location="cpu", weights_only=True)
            mel = blob["mel"]                 # (1,F,T)
            target_tf = blob["target"]        # (C,F,T)
            if self.target_type == "tf":
                return mel, target_tf
            else:
                # reduce over frequency for time-only targets
                target_time = target_tf.amax(dim=1)   # (C,T)
                return mel, target_time

        # --------- (B) on-the-fly from annotations -----------------------
        row = self.df.iloc[idx]
        fname = row["Filename"]
        win_s = float(row["Start Time (s)"])
        win_e = win_s + self.window_len

        wav, fs = torchaudio.load(os.path.join(self.data_path, fname))
        if fs != self.sr:
            wav = torchaudio.transforms.Resample(fs, self.sr)(wav)

        s0, s1 = int(win_s * self.sr), int(win_e * self.sr)
        clip = wav[:, s0:s1]
        mel = self.db_tf(self.mel_tf(clip))  # (1,F,Tvar)

        T = mel.size(-1)
        if T < self.max_frames: mel = F.pad(mel, (0, self.max_frames - T)); T = self.max_frames
        elif T > self.max_frames: mel = mel[..., :self.max_frames]; T = self.max_frames
        mel = mel.float()

        # collect overlaps
        if self.include_overlaps:
            g = self._by_file[fname]
            overlaps = g[(g["Start Time (s)"] < win_e) & (g["End Time (s)"] > win_s)]
        else:
            overlaps = pd.DataFrame([row])

        if self.target_type == "time":
            target = torch.zeros(self.C, T, dtype=torch.float32)
            for _, r in overlaps.iterrows():
                c = self.label_to_idx[r["Species eBird Code"]]
                s = max(float(r["Start Time (s)"]), win_s)
                e = min(float(r["End Time (s)"]),   win_e)
                if e <= s: continue
                sf = int(((s - win_s) * self.sr) // self.hop)
                ef = int(np.ceil(((e - win_s) * self.sr) / self.hop))
                sf = max(0, min(T, sf)); ef = max(0, min(T, ef))
                if ef > sf: target[c, sf:ef] = 1.0
            return mel, target

        elif self.target_type == "tf":
            target = torch.zeros(self.C, self.n_mels, T, dtype=torch.float32)
            # build list of boxes for painter
            boxes = []
            for _, r in overlaps.iterrows():
                boxes.append((
                    float(r["Start Time (s)"]), float(r["End Time (s)"]),
                    float(r["Low Freq (Hz)"]),  float(r["High Freq (Hz)"]),
                    r["Species eBird Code"]
                ))
            target = self._paint_tf(target, boxes, win_s, win_e, T)
            return mel, target

        else:
            raise ValueError("target_type must be 'time' or 'tf'")
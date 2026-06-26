import os
import math
import torch
import random
import json
import torchaudio
import pandas as pd
import numpy as np
import time

import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Iterable, Dict, Optional, List, Tuple

_N_FFT = 1024

 # mel <-> hz conversions
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

# ---------- Helpers ----------
def _mel_bin_centers_hz(n_mels: int, sr: int, fmin: float = 0.0, fmax: float | None = None) -> torch.Tensor:
    """Approximate mel filter center frequencies (Hz) without extra deps."""
    if fmax is None:
        fmax = sr / 2

    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_centers = np.linspace(m_min, m_max, num=n_mels, dtype=np.float64)
    hz_centers = mel_to_hz(m_centers)  # shape (n_mels,)
    return torch.tensor(hz_centers, dtype=torch.float32)

def _clip_time_window(row_start, row_end, win_start, win_len):
    """Return (win_s, win_e) window, used a fixed window starting at row_start."""
    win_s = float(win_start)
    win_e = win_s + float(win_len)
    # win_e = float(row_end)
    return win_s, win_e

def _time_to_frame_ix(seconds, sr, hop_len):
    return int(np.floor((seconds * sr) / hop_len))

def _overlap_frames(s, e, win_s, win_e, sr, hop_len, T):
    """Return integer frame span [sf, ef) for (s,e) clipped to window [win_s, win_e)."""
    s_clip = max(s, win_s)
    e_clip = min(e, win_e)
    if e_clip <= s_clip:
        return None
    sf = _time_to_frame_ix(s_clip - win_s, sr, hop_len)
    ef = int(np.ceil(((e_clip - win_s) * sr) / hop_len))
    sf = max(0, min(T, sf))
    ef = max(0, min(T, ef))
    if ef <= sf:
        return None
    return sf, ef

# ---------- dataset window info ----------
def ds_window_info(ds, idx: int):
    """
    Returns (filename, win_s, win_e) for a dataset item.
    """
    if getattr(ds, "precomputed", False) and getattr(ds, "use_manifest", False):
        rec = ds.manifest.loc[idx]
        fname = os.path.basename(rec["audio_filename"])
        return fname, float(rec["win_s"]), float(rec["win_e"])
    else:
        row = ds.df.iloc[idx]
        fname = row["Filename"]
        win_s = float(row["Start Time (s)"])
        return fname, win_s, win_s + float(ds.window_len)

def annotations_in_window(annotations_df: pd.DataFrame, filename: str, win_s: float, win_e: float,
                          fully_contained: bool = False) -> pd.DataFrame:
    """
    Filter annotations for a given (filename, [win_s, win_e]).
    fully_contained=False -> any overlap; True -> box fully inside window.
    """
    df = annotations_df[annotations_df["Filename"] == filename]
    if fully_contained:
        m = (df["Start Time (s)"] >= win_s) & (df["End Time (s)"] <= win_e)
    else:
        m = (df["Start Time (s)"] < win_e) & (df["End Time (s)"] > win_s)
    return df.loc[m].copy()

# ---------- (A) MEL-only precompute ----------
def _process_row_mel(row_id, row, recordings_dir, sr, hop_length, n_mels, max_frames, out_dir):
    filename = os.path.join(recordings_dir, row["Filename"])
    start_s = float(row["Start Time (s)"])
    # Fixed window start at annotation start; length = max_frames frames
    win_s, win_e = _clip_time_window(start_s, row["End Time (s)"], start_s, max_frames * hop_length / sr)

    try:
        waveform, file_sr = torchaudio.load(filename)
    except Exception as e:
        return f"⚠️ Skipped {filename}: {e}"

    if file_sr != sr:
        waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)

    start_sample = int(win_s * sr)
    end_sample   = int(win_e * sr)
    clip = waveform[:, start_sample:end_sample]

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=_N_FFT, hop_length=hop_length, n_mels=n_mels
    )
    db = torchaudio.transforms.AmplitudeToDB()

    mel = db(mel_tf(clip)) # (1, n_mels, Tvar)
    # pad/truncate to fixed T
    T = mel.size(-1)
    if T < max_frames:
        mel = F.pad(mel, (0, max_frames - T))
    elif T > max_frames:
        mel = mel[..., :max_frames]

    out_path = os.path.join(out_dir, f"{row_id:06d}.pt")
    torch.save(mel.to(torch.float16).contiguous(), out_path)
    return None

def precompute_mels(
    annotations_file,
    recordings_dir,
    output_dir="precomputed_mels",
    sr=32000,
    hop_length=256,
    n_mels=128,
    window_len=10.0,          # seconds, fixed window
    num_workers=None,
):
    """
    Save fp16 MEL tensors (1, n_mels, T) per CSV row_id with fixed T.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(annotations_file).reset_index().rename(columns={"index": "row_id"})
    max_frames = int(round(window_len * sr / hop_length))

    if num_workers is None:
        import multiprocessing
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"🧠 Using {num_workers} CPU workers")
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for _, row in df.iterrows():
            futures.append(ex.submit(
                _process_row_mel,
                int(row["row_id"]), row, recordings_dir, sr, hop_length, n_mels, max_frames, output_dir
            ))
        for f in tqdm(as_completed(futures), total=len(futures), desc="Precomputing MELs"):
            err = f.result()
            if err: print(err)

    print(f"✅ Saved {len(df)} MELs to {output_dir}/")

# ---------- (B) MEL + time–frequency target precompute ----------
def _choose_window_around_box(s, e, window_len, rng, file_dur, len_soft=True, sr=32000):
    """
    len_soft=True  -> fixed window_len windows
    len_soft=False -> window_len is MAX; if box shorter, use box length (variable)
    """
    L = max(0.0, e - s)

    if len_soft:
        win_len = window_len
    else:
        win_len = min(window_len, L)
        min_len_s = _N_FFT / float(sr)
        win_len = max(win_len, min_len_s)

    if L >= win_len:
        mid = 0.5 * (s + e)
        win_s = mid - 0.5 * win_len
    else:
        slack = max(0.0, win_len - L)
        win_s = s - rng.uniform(0.0, slack)

    win_s = max(0.0, min(win_s, max(0.0, file_dur - win_len)))
    return win_s, win_s + win_len


def _overlap_frames_fast(s, e, win_s, sr, hop, T):
    """
    Map absolute [s,e] to window-relative frame indices [sf,ef) in [0,T].
    """
    sf = int(math.floor((s - win_s) * sr / hop))
    ef = int(math.ceil((e - win_s) * sr / hop))
    if ef <= 0 or sf >= T:
        return None
    sf = max(0, min(sf, T))
    ef = max(0, min(ef, T))
    if ef <= sf:
        return None
    return sf, ef


def _process_file_random_windows_fast(
    filename, rows, sr, hop, n_mels, max_frames, out_dir,
    label_to_idx, hz_centers,
    window_len, seed,
    len_soft=True,
):
    # avoid CPU oversubscription inside multiprocessing
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # load and resample once per file
    try:
        wav, file_sr = torchaudio.load(filename)
    except Exception as e:
        return [], [f"⚠️ Skipped {filename}: {e}"]

    # force mono for consistent mel shape (1, n_mels, T)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if file_sr != sr:
        wav = torchaudio.transforms.Resample(file_sr, sr)(wav)

    file_dur = wav.size(-1) / sr

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=_N_FFT, hop_length=hop, n_mels=n_mels, center=False
    )
    db_tf = torchaudio.transforms.AmplitudeToDB()

    # >>> BIG SPEEDUP: compute mel for the whole file ONCE
    mel_full = db_tf(mel_tf(wav))           # (1, n_mels, T_full)
    T_full = int(mel_full.shape[-1])

    # rows -> numpy arrays once
    N = len(rows)
    starts = np.fromiter((float(r["Start Time (s)"]) for r in rows), dtype=np.float64, count=N)
    ends   = np.fromiter((float(r["End Time (s)"])   for r in rows), dtype=np.float64, count=N)
    lowhz  = np.fromiter((float(r["Low Freq (Hz)"])  for r in rows), dtype=np.float64, count=N)
    highhz = np.fromiter((float(r["High Freq (Hz)"]) for r in rows), dtype=np.float64, count=N)
    spcode = np.array([r["Species eBird Code"] for r in rows], dtype=object)
    row_id = np.fromiter((int(r["row_id"]) for r in rows), dtype=np.int64, count=N)

    # precompute mel-bin spans per row (faster than boolean masks)
    centers = np.asarray(hz_centers)
    lo_bin = np.searchsorted(centers, lowhz, side="left")
    hi_bin = np.searchsorted(centers, highhz, side="right")

    used = np.zeros(N, dtype=bool)
    rng = random.Random((seed ^ hash(os.path.basename(filename))) & 0xFFFFFFFF)

    manifest_rows, logs = [], []
    base = os.path.splitext(os.path.basename(filename))[0]
    sample_idx = 0

    for i in range(N):
        if used[i]:
            continue

        s0, e0 = float(starts[i]), float(ends[i])
        win_s, win_e = _choose_window_around_box(
            s0, e0, window_len, rng, file_dur, len_soft=len_soft, sr=sr
        )

        # Map window to frame indices into mel_full
        if len_soft:
            # fixed frame length windows: force exactly max_frames
            frame_s = int(math.floor(win_s * sr / hop))
            # clamp so we can take max_frames if possible
            frame_s = max(0, min(frame_s, max(0, T_full - max_frames)))
            frame_e = frame_s + max_frames
        else:
            frame_s = int(math.floor(win_s * sr / hop))
            frame_e = int(math.ceil(win_e * sr / hop))
            frame_s = max(0, min(frame_s, max(0, T_full - 1)))
            frame_e = max(frame_s + 1, min(frame_e, T_full))

        win_s = frame_s * hop / sr
        win_e = frame_e * hop / sr
        win_len_eff = win_e - win_s

        # >>> BIG SPEEDUP: vectorized "rows fully inside this window"
        in_window = (~used) & (starts >= win_s) & (ends <= win_e)
        idxs = np.where(in_window)[0]
        if idxs.size == 0:
            # fallback: at least include anchor row
            idxs = np.array([i], dtype=np.int64)

        used[idxs] = True
        src_ids = row_id[idxs].tolist()

        # mel slice
        mel = mel_full[..., frame_s:frame_e]        # (1, n_mels, T)
        T = int(mel.shape[-1])

        if len_soft:
            # pad if file is shorter than window at the end
            if T < max_frames:
                mel = F.pad(mel, (0, max_frames - T))
                T = max_frames
            elif T > max_frames:
                mel = mel[..., :max_frames]
                T = max_frames
        else:
            # variable; only truncate if something oversized
            if T > max_frames:
                mel = mel[..., :max_frames]
                T = max_frames

        # target tensor
        C = len(label_to_idx)
        target = torch.zeros((C, n_mels, T), dtype=torch.uint8)

        # fill targets (only boxes inside window)
        for idx in idxs:
            sp = spcode[idx]
            c = label_to_idx.get(sp, None)
            if c is None:
                continue

            span = _overlap_frames_fast(float(starts[idx]), float(ends[idx]), win_s, sr, hop, T)
            if span is None:
                continue
            sf, ef = span

            fb0 = int(lo_bin[idx])
            fb1 = int(hi_bin[idx])
            if fb1 <= fb0:
                continue
            fb0 = max(0, min(fb0, n_mels))
            fb1 = max(0, min(fb1, n_mels))
            if fb1 <= fb0:
                continue

            target[c, fb0:fb1, sf:ef] = 1

        out_blob = {
            "mel": mel.to(torch.float16).contiguous(),
            "target": target,
            "file": filename,
            "win_s": float(win_s),
            "win_e": float(win_e),
            "win_len": float(win_len_eff),
            "n_frames": int(T),
            "row_ids": src_ids,
            "len_soft": bool(len_soft),
        }
        out_name = f"{base}__{sample_idx:04d}.pt"
        torch.save(out_blob, os.path.join(out_dir, out_name))

        manifest_rows.append({
            "sample_id": f"{base}__{sample_idx:04d}",
            "audio_filename": filename,
            "win_s": win_s,
            "win_e": win_e,
            "win_len": win_len_eff,
            "n_frames": int(T),
            "row_ids_json": json.dumps(src_ids),
        })
        sample_idx += 1

    return manifest_rows, logs


def precompute_sed_samples(
    annotations_file,
    recordings_dir,
    output_dir="precomputed_sed_random",
    sr=32000,
    hop_length=256,
    n_mels=128,
    window_len=10.0,
    fmin=0.0,
    fmax=None,
    label_to_idx=None,
    seed=1234,
    num_workers=None,
    len_soft=True,
):
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    if ".csv" not in annotations_file:
        annotations_file = annotations_file + ".csv"

    df = pd.read_csv(annotations_file).reset_index().rename(columns={"index": "row_id"})
    if label_to_idx is None:
        species = sorted(df["Species eBird Code"].unique())
        label_to_idx = {s: i for i, s in enumerate(species)}

    # frames for MAX window_len
    max_frames = int(round(window_len * sr / hop_length))

    # keep your existing helper
    hz_centers = _mel_bin_centers_hz(n_mels, sr, fmin, fmax)

    groups = []
    for fname, g in df.groupby("Filename"):
        fullpath = os.path.join(recordings_dir, fname)
        groups.append((fullpath, list(g.to_dict("records"))))

    if num_workers is None:
        import multiprocessing
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    all_manifest, all_logs = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for fullpath, rows in groups:
            futures.append(ex.submit(
                _process_file_random_windows_fast,
                fullpath, rows, sr, hop_length, n_mels, max_frames, samples_dir,
                label_to_idx, hz_centers, window_len, seed,
                len_soft,
            ))
        for f in tqdm(as_completed(futures), total=len(futures), desc="Random-window SED"):
            manifest_rows, logs = f.result()
            all_manifest.extend(manifest_rows)
            all_logs.extend(logs)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    pd.DataFrame(all_manifest).to_csv(manifest_path, index=False)

    print(f"✅ Wrote {len(all_manifest)} windows to {samples_dir}/")
    print(f"🧾 Manifest: {manifest_path}")
    for m in all_logs:
        print(m)


# --------- benchmark_pipeline
def benchmark_pipeline(model, loader, device=None, batches=50, use_amp=True):
    """
    Times DataLoader fetch (CPU), H2D transfer, and forward (GPU).
    Returns a dict with avg ms/batch and batches/s.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval().to(device)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # warmup a couple of iters for stable timings
    it = iter(loader)
    for _ in range(2):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader); break
        xb = xb.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _ = model(xb, return_tf=(yb.dim()==4 if isinstance(yb, torch.Tensor) else False))

    # main timing loop
    fetch_s = h2d_s = fwd_s = 0.0
    n = 0
    it = iter(loader)
    while n < batches:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()  # fetch done

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            xb, yb = batch
        else:
            xb, yb = batch, None

        _sync(); t2 = time.perf_counter()
        xb = xb.to(device, non_blocking=True).float()
        # yb move not needed for pure forward timing
        _sync(); t3 = time.perf_counter()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            _ = model(xb, return_tf=(yb is not None and isinstance(yb, torch.Tensor) and yb.dim()==4))
        _sync(); t4 = time.perf_counter()

        fetch_s += (t1 - t0)
        h2d_s   += (t3 - t2)
        fwd_s   += (t4 - t3)
        n += 1

    n = max(n, 1)
    res = {
        "fetch_ms": 1000.0 * fetch_s / n,
        "h2d_ms":   1000.0 * h2d_s   / n,
        "fwd_ms":   1000.0 * fwd_s   / n,
        "batches_s": 1.0 / max((fetch_s + h2d_s + fwd_s) / n, 1e-9),
    }
    print(f"[bench {n} batches] fetch {res['fetch_ms']:.1f} ms | "
          f"H2D {res['h2d_ms']:.1f} ms | fwd {res['fwd_ms']:.1f} ms | "
          f"{res['batches_s']:.1f} batches/s")
    return res

# ---------- pos_weight helper ----------
def compute_pos_weight(loader, device, limit_batches=None, power=0.5, cap=(0.5, 8.0), normalize=True, global_scale=50.0):
    """
    time-reduced pos_weight:
      raw = neg/pos  (from y.amax(dim=2) if TF)
      w = raw**power (<= 1 keeps scale reasonable; 0.5 = sqrt)
      optionally normalize to mean=1
      clamp to cap
    """
    # 1) gather raw ratios from time-reduced targets
    pos_sum, total_sum, n = None, None, 0
    for _, y in loader:
        y = y.to(device).float()
        if y.dim() == 4:   # (B,C,F,T) -> time-reduced
            y_use = y.amax(dim=2)            # (B,C,T)
            pos = y_use.sum(dim=(0,2))       # (C,)
            total_per_class = y_use.shape[0] * y_use.shape[2]
        else:               # (B,C,T)
            pos = y.sum(dim=(0,2))
            total_per_class = y.shape[0] * y.shape[2]

        if pos_sum is None:
            pos_sum  = pos.clone()
            total_sum = torch.full_like(pos_sum, float(total_per_class))
        else:
            pos_sum += pos
            total_sum += float(total_per_class)

        n += 1
        if limit_batches is not None and n >= limit_batches:
            break

    if pos_sum is None:
        raise ValueError("No batches seen.")

    pos_sum = pos_sum.clamp_min(1.0)
    neg_sum = (total_sum - pos_sum).clamp_min(1.0)
    raw = (neg_sum / pos_sum)                # (C,)

    # 2) temper + normalize + cap
    w = raw.pow(power)
    if normalize:
        w = w / w.mean().clamp_min(1e-8)

    w = w * global_scale

    w = w.clamp(cap[0], cap[1]).contiguous()
    return w, raw

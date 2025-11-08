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

# ---- dataset window info ---------------------------------------------------------
def ds_window_info(ds, idx: int):
    """
    Returns (filename, win_s, win_e) for a dataset item.
    Works for:
      - precomputed random windows (ds.use_manifest=True)
      - on-the-fly / legacy precomputed
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

def plot_window_with_annotations(
    ds,
    annotations_df: pd.DataFrame,
    idx: int,
    title: str | None = None,
    fully_contained: bool = False,
):
    """
    Visualize ds[idx] mel + draw class boxes from annotations_df falling in the window.
    Uses mel-bin centers (consistent with precompute).
    """
    fname, win_s, win_e = ds_window_info(ds, idx)

    rows = annotations_in_window(annotations_df, fname, win_s, win_e, fully_contained=fully_contained)
    cols = [c for c in ["Filename","Start Time (s)","End Time (s)","Low Freq (Hz)","High Freq (Hz)","Species eBird Code"] if c in rows.columns]
    try:
        # in notebooks this shows a nice table
        display(rows[cols].sort_values(by=[c for c in ["Start Time (s)","End Time (s)"] if c in rows.columns]))
    except Exception:
        pass

    mel, _ = ds[idx]                       # (1,F,T)
    mel_np = mel.squeeze(0).detach().cpu().numpy()
    Fm, T = mel_np.shape

    # centers in Hz (F,)
    centers = _mel_bin_centers_hz(Fm, ds.sr)  # (F,)
    centers = centers.detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(mel_np, origin="lower", aspect="auto")
    ttl = title or f"{fname} | [{win_s:.2f}s, {win_e:.2f}s] (idx={idx})"
    plt.title(ttl); plt.xlabel("Frames"); plt.ylabel("Mel bins")
    ax = plt.gca()

    sr, hop = ds.sr, ds.hop
    for _, r in rows.iterrows():
        # time → frames (clipped to window)
        s = max(float(r["Start Time (s)"]), win_s)
        e = min(float(r["End Time (s)"]),   win_e)
        if e <= s: 
            continue
        sf = int(np.floor(((s - win_s) * sr) / hop))
        ef = int(np.ceil(((e - win_s) * sr) / hop))
        sf = max(0, min(T, sf)); ef = max(0, min(T, ef))
        if ef <= sf:
            continue

        # freq → mel bins via centers
        f_lo = float(r.get("Low Freq (Hz)", 0.0))
        f_hi = float(r.get("High Freq (Hz)", sr/2))
        fmask = (centers >= f_lo) & (centers <= f_hi)
        if not np.any(fmask):
            continue
        ys = np.where(fmask)[0]
        y0, y1 = int(ys.min()), int(ys.max())

        rect = patches.Rectangle((sf, y0), ef - sf, y1 - y0 + 1, fill=False, linewidth=1)
        ax.add_patch(rect)
        lbl = str(r.get("Species eBird Code","cls"))
        ax.text(sf + 1, y1, lbl, fontsize=7, va="top")

    plt.xlim(0, T); plt.ylim(0, Fm)
    plt.tight_layout(); plt.show()

    return rows  # handy for downstream use

# ---- random window finder --------------------------------------------------------
def find_random_window_with_n_samples(
    ds,
    annotations_df: pd.DataFrame,
    n: int,
    mode: str = "at_least",        # "at_least" | "exact" | "at_most"
    fully_contained: bool = False, # True => boxes must be entirely inside the window
    seed: int | None = None,
    max_tries: int = 2000,
):
    """
    Randomly sample dataset indices until one has the desired count of annotations.
    Returns: (idx, fname, win_s, win_e, rows) or (None, None, None, None, empty_df).
    Requires helpers: ds_window_info(...) and annotations_in_window(...).
    """
    import random
    rng = random.Random(seed)

    def _ok(cnt: int) -> bool:
        if mode == "exact":    return cnt == n
        if mode == "at_least": return cnt >= n
        if mode == "at_most":  return cnt <= n
        raise ValueError("mode must be one of: 'at_least', 'exact', 'at_most'")

    N = len(ds)
    if N == 0:
        return None, None, None, None, pd.DataFrame()

    tried = set()
    # random attempts
    for _ in range(min(max_tries, N * 4)):
        i = rng.randrange(N)
        if i in tried: 
            continue
        tried.add(i)

        fname, win_s, win_e = ds_window_info(ds, i)
        rows = annotations_in_window(annotations_df, fname, win_s, win_e, fully_contained=fully_contained)
        cnt  = int(rows.shape[0])
        if _ok(cnt):
            rows = rows.sort_values(by=[c for c in ["Start Time (s)","End Time (s)"] if c in rows.columns])
            return i, fname, win_s, win_e, rows

    # fallback: linear scan
    for i in range(N):
        if i in tried:
            continue
        fname, win_s, win_e = ds_window_info(ds, i)
        rows = annotations_in_window(annotations_df, fname, win_s, win_e, fully_contained=fully_contained)
        cnt  = int(rows.shape[0])
        if _ok(cnt):
            rows = rows.sort_values(by=[c for c in ["Start Time (s)","End Time (s)"] if c in rows.columns])
            return i, fname, win_s, win_e, rows

    return None, None, None, None, pd.DataFrame()

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
        sample_rate=sr, n_fft=1024, hop_length=hop_length, n_mels=n_mels
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
def _choose_window_around_box(s, e, window_len, rng, file_dur):
    L = max(0.0, e - s)
    if L >= window_len:
        # box longer than window -> center the window on the box midpoint
        mid = 0.5 * (s + e)
        win_s = mid - 0.5 * window_len
    else:
        slack = window_len - L
        win_s = s - rng.uniform(0.0, slack)
    # clamp to file bounds
    win_s = max(0.0, min(win_s, max(0.0, file_dur - window_len)))
    return win_s, win_s + window_len

def _process_file_random_windows(
    filename, rows, sr, hop, n_mels, max_frames, out_dir, label_to_idx, hz_centers,
    window_len, seed
):
    # load & (if needed) resample once per file
    try:
        wav, file_sr = torchaudio.load(filename)
    except Exception as e:
        return [], [f"⚠️ Skipped {filename}: {e}"]

    if file_sr != sr:
        wav = torchaudio.transforms.Resample(file_sr, sr)(wav)
    file_dur = wav.size(-1) / sr

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=hop, n_mels=n_mels, center=False
    )
    db_tf = torchaudio.transforms.AmplitudeToDB()

    used = np.zeros(len(rows), dtype=bool)
    rng = random.Random((seed ^ hash(os.path.basename(filename))) & 0xFFFFFFFF)

    manifest_rows, logs = [], []
    base = os.path.splitext(os.path.basename(filename))[0]
    sample_idx = 0

    # iterate until all rows consumed by some window
    for i in range(len(rows)):
        if used[i]:
            continue
        r0 = rows[i]
        s0, e0 = float(r0["Start Time (s)"]), float(r0["End Time (s)"])

        win_s, win_e = _choose_window_around_box(s0, e0, window_len, rng, file_dur)

        # collect all boxes overlapped by this window and mark them used
        boxes = []
        src_ids = []
        for j, r in enumerate(rows):
            if used[j]:
                continue
            s2, e2 = float(r["Start Time (s)"]), float(r["End Time (s)"])
            if (s2 >= win_s) and (e2 <= win_e):
                boxes.append((
                    s2, e2,
                    float(r["Low Freq (Hz)"]), float(r["High Freq (Hz)"]),
                    r["Species eBird Code"]
                ))
                used[j] = True
                src_ids.append(int(r["row_id"]))

        # slice audio (pad if window exceeds tail by a hair)
        start_sample = int(math.floor(win_s * sr))
        need = int(round(window_len * sr))
        end_sample = start_sample + need
        if end_sample > wav.size(-1):
            pad = end_sample - wav.size(-1)
            clip = F.pad(wav, (0, pad))
        else:
            clip = wav[:, start_sample:end_sample]

        # mel -> dB
        mel = db_tf(mel_tf(clip))  # (1, n_mels, Tvar)

        # fix T
        T = mel.shape[-1]
        if T < max_frames:
            mel = F.pad(mel, (0, max_frames - T))
            T = max_frames
        elif T > max_frames:
            mel = mel[..., :max_frames]
            T = max_frames

        # target
        C = len(label_to_idx)
        target = torch.zeros((C, n_mels, T), dtype=torch.uint8)
        centers = hz_centers
        for s, e, f_lo, f_hi, sp in boxes:
            c = label_to_idx.get(sp, None)
            if c is None:
                continue
            span = _overlap_frames(s, e, win_s, win_e, sr, hop, T)
            if span is None:
                continue
            sf, ef = span
            fmask = (centers >= f_lo) & (centers <= f_hi)
            if fmask.any():
                target[c, fmask, sf:ef] = 1

        # save sample
        out_blob = {
            "mel": mel.to(torch.float16).contiguous(),
            "target": target,
            "file": filename,
            "win_s": float(win_s),
            "win_e": float(win_e),
            "row_ids": src_ids,
        }
        out_name = f"{base}__{sample_idx:04d}.pt"
        torch.save(out_blob, os.path.join(out_dir, out_name))

        manifest_rows.append({
            "sample_id": f"{base}__{sample_idx:04d}",
            "audio_filename": filename,
            "win_s": win_s,
            "win_e": win_e,
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
):
    """
    Build fixed-length windows at random offsets that *contain* one unused box each,
    mark all overlapped boxes as used, and save (mel, target) + a manifest CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    if ".csv" not in annotations_file :
        annotations_file = annotations_file + ".csv"

    df = pd.read_csv(annotations_file).reset_index().rename(columns={"index": "row_id"})
    if label_to_idx is None:
        species = sorted(df["Species eBird Code"].unique())
        label_to_idx = {s: i for i, s in enumerate(species)}

    max_frames = int(round(window_len * sr / hop_length))
    hz_centers = _mel_bin_centers_hz(n_mels, sr, fmin, fmax)  # centers in Hz  

    # group rows by file
    groups = []
    for fname, g in df.groupby("Filename"):
        filename = os.path.join(recordings_dir, fname)
        groups.append((filename, list(g.to_dict("records"))))

    if num_workers is None:
        import multiprocessing
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    all_manifest, all_logs = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for filename, rows in groups:
            futures.append(ex.submit(
                _process_file_random_windows,
                filename, rows, sr, hop_length, n_mels, max_frames, samples_dir,
                label_to_idx, hz_centers, window_len, seed
            ))
        for f in tqdm(as_completed(futures), total=len(futures), desc="Random-window SED"):
            manifest_rows, logs = f.result()
            all_manifest.extend(manifest_rows)
            all_logs.extend(logs)

    # write manifest
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
def compute_pos_weight(loader, device, limit_batches=None, power=0.5, cap=(0.5, 8.0), normalize=True):
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
    w = w.clamp(cap[0], cap[1]).contiguous()
    return w, raw

# ---------- plotting utilities (Plotly) ----------
def _safe_import_plotly_pd():
    try:
        import plotly.graph_objects as go
        return pd, go
    except Exception:
        return None, None

def save_plotly_curves_from_csv(csv_path: str, html_path: str,
                                y_cols: Tuple[str, ...] = ("train_loss", "val_loss", "tf_f1_micro", "time_f1_micro")) -> bool:
    """
    Render an HTML with selected y_cols over 'epoch' from a single CSV.
    Returns True if written.
    """
    pd, go = _safe_import_plotly_pd()
    if pd is None: return False
    if not os.path.isfile(csv_path): return False

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns: return False

    fig = go.Figure()
    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["epoch"], y=df[col], name=col, mode="lines+markers"))
    fig.update_layout(title=os.path.basename(csv_path),
                      xaxis_title="epoch", template="plotly_dark",
                      width=980, height=520)
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    fig.write_html(html_path)
    return True

def load_runs_under(root: str) -> List[str]:
    """
    Find run folders recursively that contain logs/metrics.csv.
    Returns list of metrics.csv paths.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "metrics.csv" in filenames:
            found.append(os.path.join(dirpath, "metrics.csv"))
    return sorted(found)

def compare_runs_plot(runs_csv: List[str], html_path: str, metric: str = "val_loss", label_from: str = "run"):
    """
    Create a comparison HTML plot for a metric across multiple runs (CSV files).
    label_from: "run" -> parent folder name, "file" -> csv filename
    """
    pd, go = _safe_import_plotly_pd()
    if pd is None: return False
    fig = go.Figure()

    for csv_path in runs_csv:
        df = pd.read_csv(csv_path)
        if "epoch" not in df.columns or metric not in df.columns:
            continue
        if label_from == "run":
            label = os.path.basename(os.path.dirname(csv_path))  # 'logs' folder parent (run id)
        else:
            label = os.path.basename(csv_path)
        fig.add_trace(go.Scatter(x=df["epoch"], y=df[metric], name=label, mode="lines+markers"))

    fig.update_layout(title=f"Compare runs: {metric}",
                      xaxis_title="epoch", yaxis_title=metric,
                      template="plotly_dark", width=1100, height=600)
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    fig.write_html(html_path)
    return True
from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -------------------------
# Annotations I/O
# -------------------------

REQUIRED_COLS = [
    "Filename",
    "Start Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species eBird Code",
]

def load_annotations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


# -------------------------
# Recording path resolver
# -------------------------

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def resolve_recording_path(recordings_file: str, recordings_location: str) -> str:
    """
    Find recordings_file inside recordings_location.
    - tries direct path
    - tries recursive exact filename match
    - tries fuzzy match by normalized filename
    """
    root = Path(recordings_location)
    target_name = Path(recordings_file).name

    direct = root / recordings_file
    if direct.exists():
        return str(direct)

    hits = [p for p in root.rglob(target_name) if p.name == target_name]
    if hits:
        return str(hits[0])

    exts = ("*.flac", "*.wav", "*.mp3", "*.ogg", "*.m4a")
    candidates = []
    for ext in exts:
        candidates.extend(root.rglob(ext))
    if not candidates:
        raise FileNotFoundError(f"No audio files found under: {recordings_location}")

    key = _norm_name(target_name)
    cmap = {_norm_name(p.name): str(p) for p in candidates}
    if key in cmap:
        return cmap[key]

    best = difflib.get_close_matches(key, list(cmap.keys()), n=1, cutoff=0.2)
    if best:
        return cmap[best[0]]

    raise FileNotFoundError(f"Could not find '{recordings_file}' under '{recordings_location}'")


# -------------------------
# Plot helpers
# -------------------------

def _hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    hex_color = str(hex_color).lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(31,119,180,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def make_color_map(
    species_list: Sequence[str],
    palette: Sequence[str] = px.colors.qualitative.Dark24,
) -> Dict[str, str]:
    colors = list(palette) if palette else list(px.colors.qualitative.Plotly)
    cmap: Dict[str, str] = {}
    for i, sp in enumerate(species_list):
        cmap[sp] = colors[i % len(colors)]
    return cmap

def _match_filename(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Try exact match; else compare basenames."""
    d = df[df["Filename"] == filename]
    if not d.empty:
        return d
    base = Path(filename).name
    return df[df["Filename"].apply(lambda x: Path(str(x)).name) == base]


# -------------------------
# Audio window -> spectrogram (linear STFT)
# -------------------------

def compute_spectrogram_db_window(
    audio_path: str,
    t0: float,
    t1: float,
    sr: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 256,
    power: float = 2.0,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads ONLY [t0, t1] from the recording, computes a dB spectrogram.

    Returns:
      spec_db: (F, T) numpy array
      times_s: (T,) absolute times in seconds (aligned to recording timeline)
      freqs_hz: (F,) frequencies in Hz
    """
    try:
        import torchaudio
    except Exception as e:
        raise ImportError("This function requires torchaudio installed.") from e

    if t1 <= t0:
        raise ValueError(f"Invalid window: t0={t0}, t1={t1}")

    info = torchaudio.info(audio_path)
    file_sr = int(info.sample_rate)
    n_frames_total = int(info.num_frames)

    start_frame = max(0, int(round(t0 * file_sr)))
    end_frame = min(n_frames_total, int(round(t1 * file_sr)))
    num_frames = max(0, end_frame - start_frame)
    if num_frames <= 0:
        raise ValueError("Requested window is outside audio bounds (no frames).")

    wav, _ = torchaudio.load(audio_path, frame_offset=start_frame, num_frames=num_frames)  # (ch, n)
    wav = wav.mean(dim=0, keepdim=True)  # mono

    if file_sr != sr:
        wav = torchaudio.transforms.Resample(file_sr, sr)(wav)

    spec_tf = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=power, center=False
    )
    db_tf = torchaudio.transforms.AmplitudeToDB()

    spec = db_tf(spec_tf(wav))[0]  # (F, T)
    spec_np = spec.cpu().numpy()

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = t0 + np.arange(spec_np.shape[1]) * (hop_length / sr)  # absolute times

    upper = fmax if fmax is not None else float(freqs.max())
    f_mask = (freqs >= fmin) & (freqs <= upper)
    spec_np = spec_np[f_mask, :]
    freqs = freqs[f_mask]

    return spec_np, times, freqs


def _overlay_label_boxes_linear(
    fig: go.Figure,
    df_file: pd.DataFrame,
    t0: float,
    t1: float,
    color_map: Dict[str, str],
    opacity: float = 0.25,
    show_legend: bool = True,
) -> None:
    """
    Overlay TF rectangles (Hz) for labels overlapping [t0, t1] using filled polygons (hover works).
    """
    d = df_file[(df_file["End Time (s)"] > t0) & (df_file["Start Time (s)"] < t1)].copy()
    if d.empty:
        return

    d.sort_values("Start Time (s)", inplace=True)

    for _, r in d.iterrows():
        sp = r["Species eBird Code"]
        s = float(r["Start Time (s)"])
        e = float(r["End Time (s)"])
        lo = float(r["Low Freq (Hz)"])
        hi = float(r["High Freq (Hz)"])

        col = color_map.get(sp, "#1f77b4")
        fill = _hex_to_rgba(col, alpha=opacity)

        fig.add_trace(
            go.Scatter(
                x=[s, e, e, s, s],
                y=[lo, lo, hi, hi, lo],
                mode="lines",
                fill="toself",
                line=dict(color=col, width=1),
                fillcolor=fill,
                name=sp,
                showlegend=False,
                hovertemplate=(
                    f"Species: {sp}<br>"
                    f"Start: {s:.2f}s<br>"
                    f"End: {e:.2f}s<br>"
                    f"Low: {lo:.0f} Hz<br>"
                    f"High: {hi:.0f} Hz"
                    "<extra></extra>"
                ),
            )
        )

    if show_legend:
        for sp, col in color_map.items():
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color=col, width=6),
                    name=sp,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )


# -------------------------
# Main plotting: explicit time window from recording
# -------------------------

def plot_time_window(
    df: pd.DataFrame,
    recordings_file: str,
    recordings_location: str,
    t0: float,
    t1: float,
    sr: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    overlay_labels: bool = True,
    opacity: float = 0.25,
    color_map: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    height: int = 650,
) -> go.Figure:
    """
    Plot spectrogram for [t0, t1] with optional label overlay for that same window.
    """
    audio_path = resolve_recording_path(recordings_file, recordings_location)
    spec_db, times, freqs = compute_spectrogram_db_window(
        audio_path, t0=t0, t1=t1, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=spec_db,
            x=times,
            y=freqs,
            colorbar=dict(title="dB"),
        )
    )

    fig.update_layout(
        title=title or f"{Path(recordings_file).name}  |  window [{t0:.2f}, {t1:.2f}]",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        xaxis=dict(range=[t0, t1]),
        height=height,
    )

    if overlay_labels:
        df_file = _match_filename(df, recordings_file)
        species_list = sorted(df_file["Species eBird Code"].unique().tolist()) if not df_file.empty else []
        if color_map is None:
            color_map = make_color_map(species_list)
        _overlay_label_boxes_linear(fig, df_file, t0=t0, t1=t1, color_map=color_map, opacity=opacity, show_legend=True)

    return fig


def plot_window_for_sample(
    df: pd.DataFrame,
    recordings_location: str,
    sample: Union[int, pd.Series, dict],
    window_len: float = 10.0,
    mode: str = "center",  # "center" | "start"
    **plot_kwargs,
) -> go.Figure:
    """
    Plot a window for one annotation row.
    """
    if isinstance(sample, int):
        r = df.iloc[sample]
    elif isinstance(sample, pd.Series):
        r = sample
    else:
        r = pd.Series(sample)

    recordings_file = str(r["Filename"])
    s = float(r["Start Time (s)"])
    e = float(r["End Time (s)"])
    mid = 0.5 * (s + e)

    if mode == "start":
        t0 = max(0.0, s)
    elif mode == "center":
        t0 = max(0.0, mid - 0.5 * window_len)
    else:
        raise ValueError("mode must be 'center' or 'start'")

    t1 = t0 + float(window_len)

    sp = str(r.get("Species eBird Code", ""))
    title = plot_kwargs.pop("title", None)
    if title is None:
        title = f"Sample window ({sp}) | {Path(recordings_file).name} | [{t0:.2f}, {t1:.2f}]"

    return plot_time_window(
        df=df,
        recordings_file=recordings_file,
        recordings_location=recordings_location,
        t0=t0,
        t1=t1,
        title=title,
        **plot_kwargs,
    )


# -------------------------
# Precomputed samples (manifest.csv + samples/*.pt)
# -------------------------

def load_manifest(precomputed_dir: str, manifest_name: str = "manifest.csv") -> pd.DataFrame:
    path = Path(precomputed_dir) / manifest_name
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path)

def _get_sample_pt_path(precomputed_dir: str, sample_id: str, samples_subdir: str = "samples") -> Path:
    p = Path(precomputed_dir) / samples_subdir / f"{sample_id}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Sample .pt not found: {p}")
    return p

@lru_cache(maxsize=32)
def _mel_bin_centers_hz(n_mels: int, sr: int, n_fft: int, fmin: float, fmax: float) -> np.ndarray:
    """
    Approximate mel-bin center frequencies (Hz) using the max-weight frequency bin of each mel filter.
    Cached for speed across repeated plots.
    """
    try:
        import torch
        import torchaudio
    except Exception:
        # fallback
        return np.linspace(fmin, fmax, n_mels, dtype=np.float64)

    n_freqs = n_fft // 2 + 1
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=float(fmin),
        f_max=float(fmax),
        n_mels=int(n_mels),
        sample_rate=int(sr),
        norm=None,
        mel_scale="htk",
    )  # (n_freqs, n_mels)

    # frequency bin centers (Hz)
    freqs = torch.linspace(0, sr / 2, steps=n_freqs)
    # pick bin with max weight for each mel band
    idx = torch.argmax(fb, dim=0)  # (n_mels,)
    centers = freqs[idx].cpu().numpy().astype(np.float64)
    return centers

def _overlay_label_boxes_on_mel(
    fig: go.Figure,
    ann: pd.DataFrame,
    t0: float,
    t1: float,
    color_map: Dict[str, str],
    opacity: float,
    fmin: float,
    fmax: float,
    show_legend: bool = True,
) -> None:
    """
    Overlay TF rectangles (Hz bounds) on a mel heatmap whose y-axis is in Hz.
    """
    d = ann[(ann["End Time (s)"] > t0) & (ann["Start Time (s)"] < t1)].copy()
    if d.empty:
        return

    d.sort_values("Start Time (s)", inplace=True)

    for _, r in d.iterrows():
        sp = r["Species eBird Code"]
        s = float(r["Start Time (s)"])
        e = float(r["End Time (s)"])
        lo = float(r["Low Freq (Hz)"])
        hi = float(r["High Freq (Hz)"])

        # clip to plotted freq range
        lo = max(fmin, min(lo, fmax))
        hi = max(fmin, min(hi, fmax))
        if hi <= lo:
            continue

        col = color_map.get(sp, "#1f77b4")
        fill = _hex_to_rgba(col, alpha=opacity)

        fig.add_trace(
            go.Scatter(
                x=[s, e, e, s, s],
                y=[lo, lo, hi, hi, lo],
                mode="lines",
                fill="toself",
                line=dict(color=col, width=1),
                fillcolor=fill,
                name=sp,
                showlegend=False,
                hovertemplate=(
                    f"Species: {sp}<br>"
                    f"Start: {s:.2f}s<br>"
                    f"End: {e:.2f}s<br>"
                    f"Low: {lo:.0f} Hz<br>"
                    f"High: {hi:.0f} Hz"
                    "<extra></extra>"
                ),
            )
        )

    if show_legend:
        for sp, col in color_map.items():
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color=col, width=6),
                    name=sp,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

def plot_precomputed_sample(
    precomputed_dir: str,
    annotations_df: pd.DataFrame,
    sample_id: Optional[str] = None,
    manifest_idx: Optional[int] = None,
    samples_subdir: str = "samples",
    manifest_name: str = "manifest.csv",
    # these must match how you precomputed
    sr: int = 32000,
    hop_length: int = 960,
    n_fft: int = 1024,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    overlay_labels: bool = True,
    use_row_ids_if_available: bool = True,
    opacity: float = 0.25,
    color_map: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    height: int = 650,
) -> go.Figure:
    """
    Visualise one precomputed sample (.pt) referenced by manifest.csv.
    - Plots the saved mel tensor (dB) as a heatmap.
    - Optionally overlays label boxes from annotations_df for that window.

    Provide either sample_id or manifest_idx.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("plot_precomputed_sample requires torch installed.") from e

    man = load_manifest(precomputed_dir, manifest_name=manifest_name)

    if sample_id is None and manifest_idx is None:
        raise ValueError("Provide either sample_id or manifest_idx.")
    if sample_id is not None:
        rows = man[man["sample_id"] == sample_id]
        if rows.empty:
            raise ValueError(f"sample_id '{sample_id}' not found in manifest.")
        mrow = rows.iloc[0]
    else:
        if manifest_idx < 0 or manifest_idx >= len(man):
            raise IndexError(f"manifest_idx out of range: {manifest_idx}")
        mrow = man.iloc[int(manifest_idx)]
        sample_id = str(mrow["sample_id"])

    pt_path = _get_sample_pt_path(precomputed_dir, sample_id, samples_subdir=samples_subdir)
    blob = torch.load(pt_path, map_location="cpu")

    mel = blob["mel"]  # (1, n_mels, T)
    if hasattr(mel, "numpy"):
        mel_np = mel.squeeze(0).numpy()
    else:
        mel_np = np.asarray(mel).squeeze(0)

    T = int(mel_np.shape[1])
    win_s = float(blob.get("win_s", mrow.get("win_s", 0.0)))
    win_e = float(blob.get("win_e", mrow.get("win_e", win_s + T * hop_length / sr)))

    # axes
    times = win_s + np.arange(T) * (hop_length / sr)
    fmax_eff = float(fmax) if fmax is not None else float(sr / 2)

    mel_centers = _mel_bin_centers_hz(
        n_mels=int(mel_np.shape[0]),
        sr=int(sr),
        n_fft=int(n_fft),
        fmin=float(fmin),
        fmax=float(fmax_eff),
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=mel_np,
            x=times,
            y=mel_centers,
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=title or f"Precomputed sample: {sample_id} | [{win_s:.2f}, {win_e:.2f}]",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz) (mel-bin centers)",
        xaxis=dict(range=[win_s, win_e]),
        yaxis=dict(range=[float(fmin), float(fmax_eff)]),
        height=height,
    )

    if overlay_labels:
        # best: use row_ids from pt/manifest (needs annotations_df['row_id'])
        ann_file = _match_filename(annotations_df, str(blob.get("file", mrow.get("audio_filename", ""))))
        if ann_file.empty:
            # fallback: match by manifest audio_filename basename if needed
            ann_file = _match_filename(annotations_df, Path(str(mrow.get("audio_filename", ""))).name)

        ann_to_plot = ann_file

        row_ids = blob.get("row_ids", None)
        if row_ids is None and "row_ids_json" in mrow and isinstance(mrow["row_ids_json"], str):
            try:
                row_ids = json.loads(mrow["row_ids_json"])
            except Exception:
                row_ids = None

        if use_row_ids_if_available and row_ids is not None and "row_id" in annotations_df.columns:
            ann_to_plot = ann_file[ann_file["row_id"].isin(list(row_ids))].copy()
        # else: just overlap filter inside overlay function

        species_list = sorted(ann_to_plot["Species eBird Code"].unique().tolist()) if not ann_to_plot.empty else []
        if color_map is None:
            color_map = make_color_map(species_list)

        _overlay_label_boxes_on_mel(
            fig,
            ann=ann_to_plot,
            t0=win_s,
            t1=win_e,
            color_map=color_map,
            opacity=opacity,
            fmin=float(fmin),
            fmax=float(fmax_eff),
            show_legend=True,
        )

    return fig


def plot_random_precomputed_sample(
    precomputed_dir: str,
    annotations_df: pd.DataFrame,
    seed: int = 0,
    **kwargs,
) -> Tuple[go.Figure, pd.Series]:
    """
    Pick a random manifest row and plot it.
    Returns: (fig, manifest_row)
    """
    man = load_manifest(precomputed_dir, manifest_name=kwargs.get("manifest_name", "manifest.csv"))
    if man.empty:
        raise ValueError("Manifest is empty.")
    row = man.sample(n=1, random_state=seed).iloc[0]
    fig = plot_precomputed_sample(
        precomputed_dir=precomputed_dir,
        annotations_df=annotations_df,
        sample_id=str(row["sample_id"]),
        **kwargs,
    )
    return fig, row

def plot_time_window_with_model_probs(
    annotations_df,
    recordings_file: str,
    recordings_location: str,
    t0: float,
    t1: float,
    model,
    device="cuda",
    
    sr: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = 16000.0,
    # visuals
    overlay_labels: bool = True,
    opacity: float = 0.25,
    # model output labels
    idx_to_label: dict[int, str] | None = None,
    class_indices: list[int] | None = None,
    apply_softmax: bool = True,
    height: int = 750,
):
    import numpy as np
    import torch
    import torchaudio
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pathlib import Path

    # --- resolve audio path (reuse your existing resolver if present) ---
    # if you have resolve_recording_path in visualisers.py already, it will be used.
    try:
        audio_path = resolve_recording_path(recordings_file, recordings_location)
    except NameError:
        # fallback: recordings_location/recordings_file
        audio_path = str(Path(recordings_location) / recordings_file)

    if t1 <= t0:
        raise ValueError(f"Invalid window: t0={t0}, t1={t1}")

    # --- load just the window ---
    info = torchaudio.info(audio_path)
    file_sr = int(info.sample_rate)
    n_total = int(info.num_frames)

    start_frame = max(0, int(round(t0 * file_sr)))
    end_frame = min(n_total, int(round(t1 * file_sr)))
    num_frames = max(0, end_frame - start_frame)
    if num_frames <= 0:
        raise ValueError("Requested window is outside audio bounds (no frames).")

    wav, _ = torchaudio.load(audio_path, frame_offset=start_frame, num_frames=num_frames)
    wav = wav.mean(dim=0, keepdim=True)  # mono

    if file_sr != sr:
        wav = torchaudio.transforms.Resample(file_sr, sr)(wav)

    # --- mel (dB) ---
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=float(fmin),
        f_max=float(fmax) if fmax is not None else None,
        center=False,
    )
    db_tf = torchaudio.transforms.AmplitudeToDB()
    mel_db = db_tf(mel_tf(wav))          # (1, n_mels, T)
    T = int(mel_db.shape[-1])

    hop_s = hop_length / sr
    times = t0 + np.arange(T) * hop_s

    # mel bin centers in Hz (approx via mel filter maxima)
    n_freqs = n_fft // 2 + 1
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=float(fmin),
        f_max=float(fmax) if fmax is not None else float(sr / 2),
        n_mels=int(n_mels),
        sample_rate=int(sr),
        norm=None,
        mel_scale="htk",
    )  # (n_freqs, n_mels)
    freqs = torch.linspace(0, sr / 2, steps=n_freqs)
    centers_hz = freqs[torch.argmax(fb, dim=0)].cpu().numpy()

    mel_np = mel_db.squeeze(0).cpu().numpy()

    # --- run model ---
    model.eval()
    x = mel_db.unsqueeze(0).to(device)  # (B=1, 1, n_mels, T)
    lengths = torch.tensor([T], device=device)

    with torch.inference_mode():
        try:
            out = model(x, lengths=lengths)
        except TypeError:
            out = model(x)

    # find time logits/probs tensor of shape (B,C,T)
    time_out = None
    if isinstance(out, dict):
        for k in ["time_probs", "time", "time_logits", "frame_logits", "frame_probs"]:
            if k in out and isinstance(out[k], torch.Tensor) and out[k].dim() == 3:
                time_out = out[k]
                break
    elif isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor) and item.dim() == 3:
                time_out = item
                break
    elif isinstance(out, torch.Tensor) and out.dim() == 3:
        time_out = out

    if time_out is None:
        raise ValueError("Could not find a (B,C,T) time output in model output.")

    # to probs
    if apply_softmax:
        probs = torch.softmax(time_out, dim=1)
    else:
        probs = time_out

    probs = probs[0].detach().cpu().numpy()  # (C, Tp)
    C, Tp = probs.shape

    # align time axis if Tp != T (rare, but possible)
    if Tp == T:
        prob_times = times
    else:
        prob_times = np.linspace(t0, t0 + Tp * hop_s, Tp, endpoint=False)

    # choose subset of classes
    if class_indices is None:
        class_indices = list(range(C))
    probs = probs[class_indices, :]
    y_labels = []
    for ci in class_indices:
        if idx_to_label is not None and ci in idx_to_label:
            y_labels.append(idx_to_label[ci])
        else:
            y_labels.append(str(ci))

    # --- build figure: mel on top, probs below ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.03,
        subplot_titles=(
            f"Mel (dB): {Path(recordings_file).name} [{t0:.2f}, {t1:.2f}]",
            "Model time probabilities"
        ),
    )

    fig.add_trace(
        go.Heatmap(z=mel_np, x=times, y=centers_hz, colorbar=dict(title="dB")),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=probs,
            x=prob_times,
            y=y_labels,
            colorbar=dict(title="p"),
            zmin=0.0,
            zmax=1.0,
        ),
        row=2, col=1
    )

    # overlay annotation boxes on mel (optional)
    if overlay_labels:
        # match filename (exact or basename)
        df_file = annotations_df[annotations_df["Filename"] == recordings_file]
        if df_file.empty:
            base = Path(recordings_file).name
            df_file = annotations_df[annotations_df["Filename"].apply(lambda x: Path(str(x)).name) == base]

        if not df_file.empty:
            # color map per species
            species_list = sorted(df_file["Species eBird Code"].unique().tolist())
            try:
                cmap = make_color_map(species_list)  # if you have it in visualisers.py
            except NameError:
                palette = px.colors.qualitative.Dark24
                cmap = {sp: palette[i % len(palette)] for i, sp in enumerate(species_list)}

            # add polygons (hover works)
            d = df_file[(df_file["End Time (s)"] > t0) & (df_file["Start Time (s)"] < t1)].copy()
            d.sort_values("Start Time (s)", inplace=True)

            for _, r in d.iterrows():
                sp = r["Species eBird Code"]
                s = float(r["Start Time (s)"])
                e = float(r["End Time (s)"])
                lo = float(r["Low Freq (Hz)"])
                hi = float(r["High Freq (Hz)"])

                col = cmap.get(sp, "#1f77b4")
                # hex->rgba
                hc = col.lstrip("#")
                if len(hc) == 6:
                    rr, gg, bb = int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)
                    fill = f"rgba({rr},{gg},{bb},{opacity})"
                else:
                    fill = f"rgba(31,119,180,{opacity})"

                fig.add_trace(
                    go.Scatter(
                        x=[s, e, e, s, s],
                        y=[lo, lo, hi, hi, lo],
                        mode="lines",
                        fill="toself",
                        line=dict(color=col, width=2),
                        fillcolor=fill,
                        name=sp,
                        showlegend=False,
                        hovertemplate=(
                            f"Species: {sp}<br>"
                            f"Start: {s:.2f}s<br>End: {e:.2f}s<br>"
                            f"Low: {lo:.0f} Hz<br>High: {hi:.0f} Hz"
                            "<extra></extra>"
                        ),
                    ),
                    row=1, col=1
                )

            # add legend entries (one per species)
            for sp, col in cmap.items():
                fig.add_trace(
                    go.Scatter(x=[None], y=[None], mode="lines",
                               line=dict(color=col, width=6), name=sp,
                               hoverinfo="skip", showlegend=True),
                    row=1, col=1
                )

    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Class", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=height)

    return fig
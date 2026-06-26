from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------  ----------

def load_run_df(csv_path: Path | str) -> pd.DataFrame:
    """
    Load metrics.csv
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- metric naming helpers ----------

METRIC_GROUPS: Dict[str, List[str]] = {
    "losses": [
        "loss",
    ],

    "tf_f1_dice": [
        "tf_f1_micro",
        "tf_f1_macro",
        "tf_dice",
        "tf_dice_soft",
        "tf_bal_acc",
        "tf_acc",
    ],
    "tf_prec_rec": [
        "tf_prec_micro",
        "tf_rec_micro",
        "tf_prec_macro",
        "tf_rec_macro",
    ],
    "tf_fracs": [
        "tf_present_frac",
        "tf_pred_pos_frac",
        "tf_tgt_pos_frac",
    ],

    "time_f1_dice": [
        "time_f1_micro",
        "time_f1_macro",
        "time_dice",
        "time_dice_soft",
        "time_bal_acc",
        "time_acc",
    ],
    "time_prec_rec": [
        "time_prec_micro",
        "time_rec_micro",
        "time_prec_macro",
        "time_rec_macro",
    ],
    "time_fracs": [
        "time_present_frac",
        "time_pred_pos_frac",
        "time_tgt_pos_frac",
    ],
}

METRIC_LABELS: Dict[str, str] = {
    "loss": "Loss",

    "tf_f1_micro": "TF F1 micro",
    "tf_f1_macro": "TF F1 macro",
    "tf_dice": "TF Dice",
    "tf_dice_soft": "TF Dice (soft)",
    "tf_bal_acc": "TF Balanced accuracy",
    "tf_acc": "TF Accuracy",
    "tf_prec_micro": "TF Precision micro",
    "tf_rec_micro": "TF Recall micro",
    "tf_prec_macro": "TF Precision macro",
    "tf_rec_macro": "TF Recall macro",
    "tf_present_frac": "TF fraction of present classes",
    "tf_pred_pos_frac": "TF predicted positive fraction",
    "tf_tgt_pos_frac": "TF target positive fraction",

    "time_f1_micro": "TIME F1 micro",
    "time_f1_macro": "TIME F1 macro",
    "time_dice": "TIME Dice",
    "time_dice_soft": "TIME Dice (soft)",
    "time_bal_acc": "TIME Balanced accuracy",
    "time_acc": "TIME Accuracy",
    "time_prec_micro": "TIME Precision micro",
    "time_rec_micro": "TIME Recall micro",
    "time_prec_macro": "TIME Precision macro",
    "time_rec_macro": "TIME Recall macro",
    "time_present_frac": "TIME fraction of present classes",
    "time_pred_pos_frac": "TIME predicted positive fraction",
    "time_tgt_pos_frac": "TIME target positive fraction",
}


def pretty_label(suffix: str) -> str:
    return METRIC_LABELS.get(suffix, suffix)


def metric_suffixes_from_df(df: pd.DataFrame) -> List[str]:
    """
    Infer all metric suffixes present in a metrics.csv

    Example: if columns contain 'train_tf_f1_micro' and 'val_tf_f1_micro',
    this returns 'tf_f1_micro'
    """
    suffixes: set[str] = set()
    for col in df.columns:
        if col.startswith("train_"):
            suffixes.add(col[len("train_") :])
        elif col.startswith("val_"):
            suffixes.add(col[len("val_") :])
    return sorted(suffixes)


def is_prob_metric(suffix: str) -> bool:
    """
    metrics from 0 to 1(everything but loss basically)
    """
    tokens = ["f1", "prec", "rec", "dice", "acc", "frac"]
    return any(tok in suffix for tok in tokens)


# ---------- plotting funcs ----------

def plot_metric_pair(
    df: pd.DataFrame,
    suffix: str,
    ax: plt.Axes,
    show_legend: bool = True,
) -> None:
    """
    Plot train_<suffix> and val_<suffix> on a single Axes
    If only one exists, plots whatever is available
    """
    x = df["epoch"] if "epoch" in df.columns else pd.RangeIndex(len(df))

    train_col = f"train_{suffix}"
    val_col = f"val_{suffix}"

    plotted = False

    if train_col in df.columns:
        ax.plot(x, df[train_col], label="train")
        plotted = True

    if val_col in df.columns:
        ax.plot(x, df[val_col], linestyle="--", label="val")
        plotted = True

    ax.set_title(pretty_label(suffix), fontsize=9)

    if is_prob_metric(suffix):
        ax.set_ylim(0.0, 1.0)

    ax.grid(True, linestyle=":", linewidth=0.5)

    if show_legend and plotted:
        ax.legend(fontsize=8, loc="best")


def plot_metric_matrix(
    df: pd.DataFrame,
    suffixes: Sequence[str],
    out_path: Path | str,
    title: Optional[str] = None,
    ncols: int = 3,
) -> Path:
    """
    Plots a grid of subplots, one per suffix, each with train/val curves

    Saves a PNG to out_path
    """
    if not suffixes:
        return Path(out_path)

    n = len(suffixes)
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.0 * ncols, 3.0 * nrows),
        sharex=True,
    )

    # Normalize axes into 1D list
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    for i, suffix in enumerate(suffixes):
        ax = axes_flat[i]
        plot_metric_pair(df, suffix, ax=ax, show_legend=True)

    for j in range(len(suffixes), len(axes_flat)):
        axes_flat[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[plot_metric_matrix] Saved: {out_path}")
    return out_path


# ---------- generators ----------

def generate_all_metric_plots_from_df(
    df: pd.DataFrame,
    out_dir: Path | str,
    ncols: int = 3,
) -> Dict[str, List[Path]]:
    """
    Given a loaded metrics DataFrame, generates a bunch of grouped plots
    and saves them into out_dir

    Returns: dict{ group_name: [list_of_paths] }
    """
    out_dir = ensure_dir(out_dir)
    all_suffixes = set(metric_suffixes_from_df(df))

    created: Dict[str, List[Path]] = {}

    used_suffixes: set[str] = set()

    # 1) Predefined groups (losses, tf_*, time_*)
    for group_name, suffixes in METRIC_GROUPS.items():
        present = [s for s in suffixes if s in all_suffixes]
        if not present:
            continue
        fname = f"{group_name}.png"
        path = out_dir / fname
        title = group_name.replace("_", " ").upper()
        plot_metric_matrix(df, present, path, title=title, ncols=ncols)
        created[group_name] = [path]
        used_suffixes.update(present)

    # 2) Leftover metrics -> chunk into misc figures
    leftover = sorted(all_suffixes - used_suffixes)
    if leftover:
        chunk_size = 6
        misc_paths: List[Path] = []
        for i in range(0, len(leftover), chunk_size):
            chunk = leftover[i : i + chunk_size]
            fname = f"misc_{(i // chunk_size) + 1}.png"
            path = out_dir / fname
            title = f"Misc metrics #{(i // chunk_size) + 1}"
            plot_metric_matrix(df, chunk, path, title=title, ncols=ncols)
            misc_paths.append(path)
        created["misc"] = misc_paths

    return created


def generate_all_metric_plots_from_csv(
    csv_path: Path | str,
    out_dir: Path | str,
    ncols: int = 3,
) -> Dict[str, List[Path]]:
    
    df = load_run_df(csv_path)
    return generate_all_metric_plots_from_df(df, out_dir=out_dir, ncols=ncols)


def _find_metrics_csv_in_run(run_dir: Path | str) -> Tuple[Path, Path]:
    """
    Find metircs.csv
    Returns: (csv_path, logs_dir)
    """
    run_dir = Path(run_dir)

    # <run_dir>/logs/metrics.csv
    logs_dir = run_dir / "logs"
    csv_path = logs_dir / "metrics.csv"
    if csv_path.is_file():
        return csv_path, logs_dir

    # <run_dir>/metrics.csv directly
    csv_path = run_dir / "metrics.csv"
    if csv_path.is_file():
        return csv_path, run_dir

    raise FileNotFoundError(
        f"Could not find metrics.csv in run_dir={run_dir}. "
        "Looked for 'logs/metrics.csv' and 'metrics.csv'."
    )


def generate_all_plots_for_run(
    run_dir: Path | str,
    stats_subdir: str = "stats",
    ncols: int = 3,
) -> Dict[str, List[Path]]:
    csv_path, logs_dir = _find_metrics_csv_in_run(run_dir)
    stats_dir = logs_dir / stats_subdir
    print(f"[generate_all_plots_for_run] metrics_csv={csv_path}")
    print(f"[generate_all_plots_for_run] stats_dir={stats_dir}")
    return generate_all_metric_plots_from_csv(csv_path, stats_dir, ncols=ncols)

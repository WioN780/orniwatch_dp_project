import os, json, math, argparse, time, random
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, List, Type

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

import contextlib

import time
from tqdm import tqdm

from common.logging_extensions import *
from classes.BirdDataset import BirdDataset
from common.metrics import compute_tf_metrics, compute_time_metrics

from common.common_utils import compute_pos_weight

from classes import BirdModels


# ---------- utils ----------
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def instantiate_model(model_name: str, n_mels: int, n_classes: int) -> nn.Module:
    if not hasattr(BirdModels, model_name):
        raise ValueError(f"Model class '{model_name}' not found in classes.BirdModels")
    cls = getattr(BirdModels, model_name)
    # Only pass n_mels and n_classes
    sig = cls.__init__.__code__.co_varnames
    kwargs = {}
    if "n_mels" in sig: kwargs["n_mels"] = n_mels
    if "n_classes" in sig: kwargs["n_classes"] = n_classes
    return cls(**kwargs)


# ---------- config ----------
@dataclass
class TrainConfig:
    location: str
    data_folder: str
    annotations_file: str
    precomputed: bool = True
    precomputed_dirs: Optional[List[str]] = None  # list of folders
    n_mels: int = 128
    window_len: float = 10.0
    batch_size: int = 8
    num_workers: int = 4
    target_type: str = "tf"            # "tf" or "time"
    include_overlaps: bool = True
    lr: float = 1e-3
    epochs: int = 10
    model_name: str = "BirdCRNN"
    model: Type = field(default=BirdModels.BirdCRNN)
    amp: str = "bf16"                  # "bf16", "fp16", "off"
    pos_weight: str = "auto"           # "auto" or "none"
    save_top_k: int = 3
    monitor: str = "val_f1"
    monitor_mode: str = "max"
    seed: int = 42
    logs_base: str = "logs_folder"
    run_name: Optional[str] = None


# ---------- data ----------
def create_dataloaders(cfg: TrainConfig, label_to_idx: Dict[str,int]) -> Tuple[DataLoader, DataLoader]:
    dirs = cfg.precomputed_dirs or [None]

    # train = concat of all dirs with split="train"
    train_sets: List[BirdDataset] = []
    for d in dirs:
        train_sets.append(
            BirdDataset(
                data_path=cfg.location + cfg.data_folder,
                annotations_file=cfg.location + cfg.annotations_file,
                split="train",
                precomputed=(d is not None) or cfg.precomputed,
                precomputed_dir=cfg.location + d,
                label_to_idx=label_to_idx,
                n_mels=cfg.n_mels,
                window_len=cfg.window_len,
                target_type=cfg.target_type,
                include_overlaps=cfg.include_overlaps,
            )
        )
    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]

    # val = first dir only with split="val"
    val_dir = dirs[0]
    val_ds = BirdDataset(
        data_path=cfg.location + cfg.data_folder,
        annotations_file=cfg.location + cfg.annotations_file,
        split="val",
        precomputed=(val_dir is not None) or cfg.precomputed,
        precomputed_dir=cfg.location + val_dir,
        label_to_idx=label_to_idx,
        n_mels=cfg.n_mels,
        window_len=cfg.window_len,
        target_type=cfg.target_type,
        include_overlaps=cfg.include_overlaps,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    return train_loader, val_loader


# ---------- train/eval ----------
def train_one_epoch(
    model,
    loader,
    optimizer,
    bce_time,
    bce_tf,
    device,
    amp_dtype,
    lambda_time: float,
    target_type: str,
    metric_thr: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns:
      train_loss_avg: float
      train_metrics_avg: dict with keys like train_tf_f1_macro, train_time_f1_macro, ...
    """
    model.train()
    use_amp = (device.type == "cuda") and (amp_dtype is not None)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    loss_sum, n_batches = 0.0, 0
    train_metrics_sum: Dict[str, float] = {}

    bar = tqdm(loader, desc="train", leave=False)
    for xb, yb in bar:
        t_batch0 = time.perf_counter()

        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        want_tf = (target_type == "tf" and yb.dim() == 4)

        # reset peak mem for this iteration
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # ---- forward
        t0 = time.perf_counter()
        ctx = torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        with ctx:
            time_logits, tf_logits = model(xb, return_tf=want_tf)
        t1 = time.perf_counter()

        # ---- loss
        if want_tf:
            loss_tf   = bce_tf(tf_logits, yb)                  # (B,C,F,T)
            loss_time = bce_time(time_logits, yb.amax(dim=2))  # (B,C,T)
            loss = loss_tf + lambda_time * loss_time
        else:
            loss = bce_time(time_logits, yb)
        t2 = time.perf_counter()

        # ---- backward/step
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        t3 = time.perf_counter()

        # timings + memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_cur  = torch.cuda.memory_allocated() / (1024**2)
            mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            mem_cur = mem_peak = 0.0

        # ---- quick train metrics (detach; keep it cheap)
        with torch.no_grad():
            if want_tf:
                y_time = yb.amax(dim=2)
                m_tf   = compute_tf_metrics(tf_logits.detach().float(),    yb.float(),    thr=metric_thr)
                m_time = compute_time_metrics(time_logits.detach().float(), y_time.float(), thr=metric_thr)
                m = {**m_tf, **m_time}
            else:
                m = compute_time_metrics(time_logits.detach().float(), yb.float(), thr=metric_thr)

        # aggregate (vector -> scalar mean)
        for k, v in m.items():
            if torch.is_tensor(v):
                v = float(v.mean().item()) if v.numel() > 1 else float(v.item())
            else:
                v = float(v)
            train_metrics_sum[k] = train_metrics_sum.get(k, 0.0) + v

        loss_sum += float(loss.item())
        n_batches += 1

        # tqdm bar
        bar.set_postfix({
            "loss": f"{loss_sum / n_batches:.4f}",
            "fw(ms)": f"{(t1 - t0) * 1e3:.1f}",
            "ls(ms)": f"{(t2 - t1) * 1e3:.1f}",
            "bw+opt(ms)": f"{(t3 - t2) * 1e3:.1f}",
            "mem(MB)": f"{mem_cur:.0f}",
            "peak(MB)": f"{mem_peak:.0f}",
            "iter(ms)": f"{(t3 - t_batch0) * 1e3:.1f}",
        })

    train_metrics_avg = {f"train_{k}": v / max(n_batches, 1) for k, v in train_metrics_sum.items()}
    return loss_sum / max(n_batches, 1), train_metrics_avg


@torch.no_grad()
def evaluate(
    model,
    loader,
    bce_time,
    bce_tf,
    device,
    amp_dtype,
    target_type: str,
    metric_thr: float = 0.05,
) -> Dict[str, float]:
    model.eval()
    use_amp = (device.type == "cuda") and (amp_dtype is not None)

    loss_sum, n_batches = 0.0, 0
    agg: Dict[str, float] = {}
    have_tf = False

    bar = tqdm(loader, desc="valid", leave=False)
    for xb, yb in bar:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        want_tf = (target_type == "tf" and yb.dim() == 4)

        ctx = torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        with ctx:
            time_logits, tf_logits = model(xb, return_tf=want_tf)
            if want_tf:
                have_tf = True
                y_time = yb.amax(dim=2)
                loss = bce_tf(tf_logits, yb) + 0.3 * bce_time(time_logits, y_time)
                m_tf   = compute_tf_metrics(tf_logits.float(),  yb.float(),   thr=metric_thr)
                m_time = compute_time_metrics(time_logits.float(), y_time.float(), thr=metric_thr)
                m = {**m_tf, **m_time}
            else:
                loss = bce_time(time_logits, yb)
                m = compute_time_metrics(time_logits.float(), yb.float(), thr=metric_thr)

        # aggregate (vector -> mean)
        for k, v in m.items():
            if torch.is_tensor(v):
                v = float(v.mean().item()) if v.numel() > 1 else float(v.item())
            else:
                v = float(v)
            agg[k] = agg.get(k, 0.0) + v

        loss_sum += float(loss.item()); n_batches += 1

    out = {f"val_{k}": v / max(n_batches, 1) for k, v in agg.items()}
    out["val_loss"] = loss_sum / max(n_batches, 1)

    # convenience aliases (so your CSV stays simple)
    if have_tf:
        out["val_f1"]       = out.get("val_tf_f1_macro", float("nan"))
        out["val_f1_macro"] = out.get("val_tf_f1_macro", float("nan"))
        out["val_prec"]     = out.get("val_tf_prec_macro", float("nan"))
        out["val_rec"]      = out.get("val_tf_rec_macro", float("nan"))
        out["time_f1"]      = out.get("val_time_f1_macro", float("nan"))
        out["tf_f1"]        = out.get("val_tf_f1_macro", float("nan"))
    else:
        out["val_f1"]       = out.get("val_time_f1_macro", float("nan"))
        out["val_f1_macro"] = out.get("val_time_f1_macro", float("nan"))
        out["val_prec"]     = out.get("val_time_prec_macro", float("nan"))
        out["val_rec"]      = out.get("val_time_rec_macro", float("nan"))
        out["time_f1"]      = out.get("val_time_f1_macro", float("nan"))
        out["tf_f1"]        = float("nan")

    return out

# ---------- fit ----------
def fit(cfg: TrainConfig, label_to_idx: Dict[str,int]):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dirs + logger
    run_name = cfg.run_name or now_ts()
    paths = make_run_dirs(cfg.logs_base, run_name, cfg.annotations_file)
    logger = setup_logger(paths["log_file"], name=f"trainer_{run_name}")
    attach_csv_handler(logger, paths["csv_file"], header=[
        "epoch",
        "train_loss",
        "val_loss",
        "val_f1","val_f1_macro","val_prec","val_rec","time_f1","tf_f1",
        "train_tf_f1_macro","train_tf_prec_macro","train_tf_rec_macro","train_tf_dice",
        "train_time_f1_macro","train_time_prec_macro","train_time_rec_macro","train_time_dice",
    ])

    # n_classes
    cfg.n_classes = len(label_to_idx)

    # model
    model = cfg.model.to(device)
    model.to(memory_format=torch.channels_last)

    # data
    train_loader, val_loader = create_dataloaders(cfg, label_to_idx)
    logger.info(f"train samples: {len(train_loader.dataset)}, val samples: {len(val_loader.dataset)}")

    # pos_weight
    pos_weight = None
    if cfg.pos_weight == "auto" and compute_pos_weight is not None:
        pos_weight, _ = compute_pos_weight(train_loader, device, limit_batches=None)
        logger.info(f"pos_weight(auto): {pos_weight}")

    # losses
    if pos_weight is not None:
        pos_weight_tf   = pos_weight.view(1, -1, 1, 1)  # (1,C,1,1) for TF
        pos_weight_time = pos_weight.view(1, -1, 1)     # (1,C,1)   for time
    else:
        pos_weight_tf = pos_weight_time = None

    bce_tf   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tf)
    bce_time = nn.BCEWithLogitsLoss(pos_weight=pos_weight_time)

    # optim/amp
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[cfg.amp]

    # save config
    cfg_dict = asdict(cfg)
    cfg_dict["paths"] = paths
    cfg_dict["model"] = model.__class__.__name__
    logger.info(f"config: {json.dumps(cfg_dict, indent=2, default=str)}")

    ckpts = CheckpointManager(paths["ckpt_dir"], mode=cfg.monitor_mode, k=cfg.save_top_k, filename_prefix=cfg.model_name.lower())
    monitor_key = cfg.monitor if cfg.monitor in ("val_f1","time_f1","tf_f1","val_loss") else "val_f1"

    for epoch in range(1, cfg.epochs+1):
        t0 = time.time()

        # ---- start memory history for this epoch
        mem_hist_enabled = False
        if torch.cuda.is_available() and hasattr(torch.cuda.memory, "_record_memory_history"):
            try:
                torch.cuda.memory._record_memory_history(max_entries=1_000_000)
                mem_hist_enabled = True
                logger.info(f"[epoch {epoch}] CUDA memory history: ENABLED")
            except Exception as e:
                logger.error(f"[epoch {epoch}] Failed to enable memory history: {e}")

        # ---- train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, bce_time, bce_tf, device, amp_dtype,
            lambda_time=0.3, target_type=cfg.target_type, metric_thr=0.05
        )
        # ---- validate
        val_metrics = evaluate(
            model, val_loader, bce_time, bce_tf, device, amp_dtype,
            target_type=cfg.target_type, metric_thr=0.05
        )

        # ---- dump memory snapshot and stop recording
        if mem_hist_enabled:
            snap_path = os.path.join(paths["logs_dir"], f"cuda_mem_snapshot_e{epoch:03d}.pickle")
            try:
                torch.cuda.memory._dump_snapshot(snap_path)
                logger.info(f"[epoch {epoch}] CUDA memory snapshot saved: {snap_path}")
            except Exception as e:
                logger.error(f"[epoch {epoch}] Failed to capture memory snapshot {e}")
            # stop recording for this epoch
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception as e:
                logger.error(f"[epoch {epoch}] Failed to stop memory history: {e}")

        elapsed = time.time() - t0

        # score
        score = val_metrics.get(
            monitor_key,
            -val_metrics["val_loss"] if cfg.monitor_mode == "max" else val_metrics["val_loss"]
        )

        # helper
        def get(d, k): 
            v = d.get(k, float("nan"))
            try:
                return round(float(v), 6)
            except Exception:
                return float("nan")

        # CSV row
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   get(val_metrics, "val_loss"),
            "val_f1":     get(val_metrics, "val_f1"),
            "val_f1_macro": get(val_metrics, "val_f1_macro"),
            "val_prec":   get(val_metrics, "val_prec"),
            "val_rec":    get(val_metrics, "val_rec"),
            "time_f1":    get(val_metrics, "time_f1"),
            "tf_f1":      get(val_metrics, "tf_f1"),

            # train
            "train_tf_f1_macro":   get(train_metrics, "train_tf_f1_macro"),
            "train_tf_prec_macro": get(train_metrics, "train_tf_prec_macro"),
            "train_tf_rec_macro":  get(train_metrics, "train_tf_rec_macro"),
            "train_tf_dice":       get(train_metrics, "train_tf_dice"),
            "train_time_f1_macro":   get(train_metrics, "train_time_f1_macro"),
            "train_time_prec_macro": get(train_metrics, "train_time_prec_macro"),
            "train_time_rec_macro":  get(train_metrics, "train_time_rec_macro"),
            "train_time_dice":       get(train_metrics, "train_time_dice"),
        }

        log_metrics(logger, row)
        logger.info(f"epoch {epoch:03d} | {row} | {elapsed:.1f}s")

        if ckpts.step(epoch, float(score), model.state_dict(), optimizer.state_dict()):
            logger.info(f"checkpoint improved on '{monitor_key}' -> saved (score={float(score):.6f})")

    logger.info("training done.")
    return paths


# ---------- JSON config loader ----------
def load_train_config(json_path: str) -> Tuple[TrainConfig, Dict[str, int]]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        J = json.load(f)

    # Accept list or comma string for precomputed_dirs
    pre_dirs = J.get("precomputed_dirs")
    if isinstance(pre_dirs, str):
        pre_dirs = [s.strip() for s in pre_dirs.split(",") if s.strip()]
    if pre_dirs is not None and not isinstance(pre_dirs, list):
        raise ValueError("precomputed_dirs must be a list of strings or a comma-separated string.")

    # Build cfg (model instance added after we know n_classes)
    cfg = TrainConfig(
        location          = J["location"],
        data_folder       = J["data_folder"],
        annotations_file  = J["annotations_file"],
        precomputed       = J.get("precomputed", bool(pre_dirs)),
        precomputed_dirs  = pre_dirs,
        n_mels            = J.get("n_mels", 128),
        window_len        = J.get("window_len", 10.0),
        batch_size        = J.get("batch_size", 8),
        num_workers       = J.get("num_workers", 4),
        target_type       = J.get("target_type", "tf"),
        include_overlaps  = J.get("include_overlaps", True),
        lr                = J.get("lr", 1e-3),
        epochs            = J.get("epochs", 10),
        model_name        = J.get("model_name", "BirdCRNN"),
        model             = BirdModels.BirdCRNN,   # temp
        amp               = J.get("amp", "bf16"),
        pos_weight        = J.get("pos_weight", "auto"),
        save_top_k        = J.get("save_top_k", 3),
        monitor           = J.get("monitor", "val_f1"),
        monitor_mode      = J.get("monitor_mode", "max"),
        seed              = J.get("seed", 42),
        logs_base         = J.get("logs_base", "logs_folder"),
        run_name          = J.get("run_name"),
    )

    # load label encodings to get n_classes
    enc_path = os.path.join(cfg.data_folder, "label_encodings.json")
    with open(enc_path, "r", encoding="utf-8") as f:
        encodings = json.load(f)
    label_to_idx = {k: int(v) if isinstance(v, str) and v.isdigit() else int(v)
                    for k, v in encodings["label_to_idx"].items()}
    n_classes = len(label_to_idx)

    #iInstantiate model with n_mels + n_classes
    model_instance = instantiate_model(cfg.model_name, cfg.n_mels, n_classes)
    cfg.model = model_instance

    # run name gen
    if not cfg.run_name:
        cfg.run_name = now_ts()

    return cfg, label_to_idx

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c", "--config", "--config_path",
        dest="config_path",
        type=str,
        default="train_config.json",
        help="Path to JSON train config (overrides TRAIN_CONFIG env)."
    )
    return p.parse_args()

# ---------- entrypoint ----------
if __name__ == "__main__":
    args = parse_cli()
    config_path = args.config_path or os.environ.get("TRAIN_CONFIG") or "train_config.json"
    print(f"[INFO] Using config: {config_path}")

    cfg, label_to_idx = load_train_config(config_path)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fit(cfg, label_to_idx)
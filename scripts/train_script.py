import os, json, math, argparse, time, random
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, List, Type

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

import contextlib

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
def train_one_epoch(model, loader, optimizer, bce_time, bce_tf, device, amp_dtype, lambda_time: float, target_type: str):
    model.train()
    use_amp = (device.type == "cuda") and (amp_dtype is not None)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    loss_sum, n_batches = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()

        want_tf = (target_type == "tf" and yb.dim() == 4)

        optimizer.zero_grad(set_to_none=True)
        ctx = torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        with ctx:
            time_logits, tf_logits = model(xb, return_tf=want_tf)

            if want_tf:

                if n_batches == 0:

                    y_tf   = yb
                    y_time = yb.amax(dim=2)     

                    print("\n[DEBUG] tf_logits mean/std:", tf_logits.detach().float().mean().item(),
                        tf_logits.detach().float().std().item())
                    print("[DEBUG] y_tf mean/std:", y_tf.float().mean().item(), y_tf.float().std().item())
                    print("[DEBUG] time_logits mean/std:", time_logits.detach().float().mean().item(),
                        time_logits.detach().float().std().item())
                    print("[DEBUG] y_time mean/std:", y_time.float().mean().item(), y_time.float().std().item())

                # TF head
                loss_tf = bce_tf(tf_logits, yb)          # (B,C,F,T)
                # TIME head from TF target
                loss_time = bce_time(time_logits, yb.amax(dim=2))  # (B,C,T)
                loss = loss_tf + lambda_time * loss_time
            else:
                loss = bce_time(time_logits, yb)         # (B,C,T)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()

        loss_sum += float(loss.item()); n_batches += 1

    return loss_sum / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, bce_time, bce_tf, device, amp_dtype, target_type: str) -> Dict[str, float]:
    model.eval()
    use_amp = (device.type == "cuda") and (amp_dtype is not None)

    loss_sum, n_batches = 0.0, 0
    agg: Dict[str, float] = {}
    have_tf = False

    for xb, yb in loader:
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
                m_tf   = compute_tf_metrics(tf_logits, yb, thr=0.05)
                m_time = compute_time_metrics(time_logits, y_time, thr=0.05)
                m = {**m_tf, **m_time}
            else:
                loss = bce_time(time_logits, yb)
                m = compute_time_metrics(time_logits, yb, thr=0.05)

        for k, v in m.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    val = float(v.item())
                else:
                    val = float(v.mean().item())
            else:
                val = float(v)
            agg[k] = agg.get(k, 0.0) + val
        loss_sum += float(loss.item()); n_batches += 1

    out = {k: v / max(n_batches, 1) for k, v in agg.items()}
    out["val_loss"] = loss_sum / max(n_batches, 1)

    if have_tf:
        out["val_f1"] = out.get("tf_f1_macro", float("nan"))
        out["val_f1_macro"] = out.get("tf_f1_macro", float("nan"))
        out["val_prec"] = out.get("tf_prec_macro", float("nan"))
        out["val_rec"] = out.get("tf_rec_macro", float("nan"))
    else:
        out["val_f1"] = out.get("time_f1_macro", float("nan"))
        out["val_f1_macro"] = out.get("time_f1_macro", float("nan"))
        out["val_prec"] = out.get("time_prec_macro", float("nan"))
        out["val_rec"] = out.get("time_rec_macro", float("nan"))

    out["time_f1"] = out.get("time_f1_macro", float("nan"))
    out["tf_f1"]   = out.get("tf_f1_macro", float("nan"))
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
        "epoch","train_loss","val_loss","val_f1","val_f1_macro","val_prec","val_rec","time_f1","tf_f1"
    ])

    # n_classes from labels
    cfg.n_classes = len(label_to_idx)

    # model
    model = cfg.model.to(device)
    model.to(memory_format=torch.channels_last)

    # data
    train_loader, val_loader = create_dataloaders(cfg, label_to_idx)

    print(f"[DEBUG] train samples: {len(train_loader.dataset)}, val samples: {len(val_loader.dataset)}")

    # optional pos_weight
    pos_weight = None
    if cfg.pos_weight == "auto" and compute_pos_weight is not None:
        pos_weight, raw_ratio = compute_pos_weight(train_loader, device, limit_batches=None)
        logger.info(f"pos_weight(auto): {pos_weight}")

    # loss/optim/amp
    if pos_weight is not None:
        pos_weight_tf   = pos_weight.view(1, -1, 1, 1)  # for (B,C,F,T)
        pos_weight_time = pos_weight.view(1, -1, 1)     # for (B,C,T)
    else:
        pos_weight_tf = pos_weight_time = None

    bce_tf   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tf)
    bce_time = nn.BCEWithLogitsLoss(pos_weight=pos_weight_time)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
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
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_time, bce_tf, device, amp_dtype, lambda_time=0.3, target_type=cfg.target_type)
        val_metrics = evaluate(model, val_loader, bce_time, bce_tf, device, amp_dtype, cfg.target_type)
        elapsed = time.time() - t0

        score = val_metrics.get(monitor_key, -val_metrics["val_loss"] if cfg.monitor_mode=="max" else val_metrics["val_loss"])
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics.get("val_loss", float("nan")), 6),
            "val_f1": round(val_metrics.get("val_f1", float("nan")), 6),
            "val_f1_macro": round(val_metrics.get("val_f1_macro", float("nan")), 6),
            "val_prec": round(val_metrics.get("val_prec", float("nan")), 6),
            "val_rec": round(val_metrics.get("val_rec", float("nan")), 6),
            "time_f1": round(val_metrics.get("time_f1", float("nan")), 6),
            "tf_f1": round(val_metrics.get("tf_f1", float("nan")), 6),
        }
        log_metrics(logger, row)
        logger.info(f"epoch {epoch:03d} | {row} | {elapsed:.1f}s")

        if ckpts.step(epoch, float(score), model.state_dict(), optimizer.state_dict()):
            logger.info(f"checkpoint improved on '{monitor_key}' -> saved (score={float(score):.6f})")

    logger.info("training done.")
    return paths


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--location", type=str, required=True)
    p.add_argument("--data_folder", type=str, required=True)
    p.add_argument("--annotations_file", type=str, required=True)
    p.add_argument("--precomputed_dirs", type=str, nargs="*", default=None,
                   help="List of precomputed feature dirs. Train uses all; val uses the first.")
    p.add_argument("--logs_base", type=str, default="logs_folder")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--model_name", type=str, default="BirdCRNN",
                   help="Class name from classes.BirdModels (e.g., BirdCRNN)")

    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--target_type", type=str, default="tf", choices=["tf","time"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", type=str, default="bf16", choices=["bf16","fp16","off"])
    p.add_argument("--pos_weight", type=str, default="auto", choices=["auto","none"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pre_dirs = args.precomputed_dirs
    if isinstance(pre_dirs, list) and len(pre_dirs) == 1 and ("," in pre_dirs[0]):
        pre_dirs = [s.strip() for s in pre_dirs[0].split(",") if s.strip()]

    cfg = TrainConfig(
        location=args.location,
        data_folder=args.data_folder,
        annotations_file=args.annotations_file,
        precomputed=True if pre_dirs else False,
        precomputed_dirs=pre_dirs,
        n_mels=args.n_mels,
        target_type=args.target_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        amp=args.amp,
        pos_weight=("auto" if args.pos_weight=="auto" else "none"),
        model_name=args.model_name,
        model=instantiate_model(args.model_name, args.n_mels, args.n_classes),
        seed=args.seed,
        logs_base=args.logs_base,
        run_name=args.run_name or now_ts(),
    )

    enc_path = os.path.join(args.data_folder, "label_encodings.json")
    with open(enc_path, "r", encoding="utf-8") as f:
        encodings = json.load(f)
    label_to_idx = {k: int(v) if isinstance(v, str) and v.isdigit() else int(v) for k, v in encodings["label_to_idx"].items()}

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fit(cfg, label_to_idx)
import torch

EPS = 1e-8  # numerical stability


# ---------- utils ----------
def binarize(p: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (p > thr).to(p.dtype)


# ---------- dice ----------
def soft_dice_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft (probability) Dice, micro-averaged over all dims."""
    p = logits.sigmoid()
    inter = (p * target).sum()
    denom = p.sum() + target.sum()
    return (2 * inter + EPS) / (denom + EPS)


def hard_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Hard (thresholded) Dice, micro-averaged."""
    p_bin = binarize(logits.sigmoid(), thr)
    inter = (p_bin * target).sum()
    denom = p_bin.sum() + target.sum()
    return (2 * inter + EPS) / (denom + EPS)


# ---------- precision / recall / F1 ----------
def prf1_micro_from_logits(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    """Micro P/R/F1 (aggregate all elements)."""
    p_bin = binarize(logits.sigmoid(), thr)
    tp = (p_bin * target).sum()
    fp = (p_bin * (1 - target)).sum()
    fn = ((1 - p_bin) * target).sum()
    prec = (tp + EPS) / (tp + fp + EPS)
    rec  = (tp + EPS) / (tp + fn + EPS)
    f1   = (2 * prec * rec + EPS) / (prec + rec + EPS)
    return prec, rec, f1


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Plain accuracy (can be dominated by background)."""
    p_bin = binarize(logits.sigmoid(), thr)
    return (p_bin == target).to(torch.float32).mean()


# ---------- macro (per-class) ----------
def prf1_macro_from_logits(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    """
    Macro P/R/F1 (average per-class). Works for (B,C,T) or (B,C,F,T).
    Returns (prec_macro, rec_macro, f1_macro, f1_per_class).
    """
    p_bin = binarize(logits.sigmoid(), thr)
    C = target.shape[1]
    t_flat = target.reshape(target.shape[0], C, -1).transpose(0, 1).reshape(C, -1)
    p_flat = p_bin.reshape(p_bin.shape[0], C, -1).transpose(0, 1).reshape(C, -1)

    tp = (p_flat * t_flat).sum(dim=1)
    fp = (p_flat * (1 - t_flat)).sum(dim=1)
    fn = ((1 - p_flat) * t_flat).sum(dim=1)

    prec = (tp + EPS) / (tp + fp + EPS)
    rec  = (tp + EPS) / (tp + fn + EPS)
    f1   = (2 * prec * rec + EPS) / (prec + rec + EPS)
    return prec.mean(), rec.mean(), f1.mean(), f1


def balanced_accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    """Macro-averaged balanced accuracy = (TPR+TNR)/2 across classes."""
    p_bin = binarize(logits.sigmoid(), thr)
    C = target.shape[1]
    t_flat = target.reshape(target.shape[0], C, -1).transpose(0, 1).reshape(C, -1)
    p_flat = p_bin.reshape(p_bin.shape[0], C, -1).transpose(0, 1).reshape(C, -1)

    tp = (p_flat * t_flat).sum(dim=1)
    tn = ((1 - p_flat) * (1 - t_flat)).sum(dim=1)
    fp = (p_flat * (1 - t_flat)).sum(dim=1)
    fn = ((1 - p_flat) * t_flat).sum(dim=1)

    tpr = (tp + EPS) / (tp + fn + EPS)  # recall
    tnr = (tn + EPS) / (tn + fp + EPS)  # specificity
    bal = (tpr + tnr) / 2
    return bal.mean(), tpr.mean(), tnr.mean()


# ---------- packs ----------
def compute_tf_metrics(tf_logits: torch.Tensor, y_tf: torch.Tensor, thr: float = 0.5) -> dict:
    """For TF supervision: logits/target (B,C,F,T)."""
    prec_mi, rec_mi, f1_mi = prf1_micro_from_logits(tf_logits, y_tf, thr)
    prec_ma, rec_ma, f1_ma, f1_vec = prf1_macro_from_logits(tf_logits, y_tf, thr)
    bal_acc, _, _ = balanced_accuracy_from_logits(tf_logits, y_tf, thr)
    return {
        "tf_dice_soft": soft_dice_from_logits(tf_logits, y_tf),
        "tf_dice":      hard_dice_from_logits(tf_logits, y_tf, thr),
        "tf_prec_micro": prec_mi, "tf_rec_micro": rec_mi, "tf_f1_micro": f1_mi,
        "tf_prec_macro": prec_ma, "tf_rec_macro": rec_ma, "tf_f1_macro": f1_ma,
        "tf_bal_acc":    bal_acc,
        "tf_acc":        accuracy_from_logits(tf_logits, y_tf, thr),  # optional
        "tf_f1_per_class": f1_vec,  # (C,)
    }


def compute_time_metrics(time_logits: torch.Tensor, y_time: torch.Tensor, thr: float = 0.5) -> dict:
    """For time supervision: logits/target (B,C,T)."""
    prec_mi, rec_mi, f1_mi = prf1_micro_from_logits(time_logits, y_time, thr)
    prec_ma, rec_ma, f1_ma, f1_vec = prf1_macro_from_logits(time_logits, y_time, thr)
    bal_acc, _, _ = balanced_accuracy_from_logits(time_logits, y_time, thr)
    return {
        "time_dice_soft": soft_dice_from_logits(time_logits, y_time),
        "time_dice":      hard_dice_from_logits(time_logits, y_time, thr),
        "time_prec_micro": prec_mi, "time_rec_micro": rec_mi, "time_f1_micro": f1_mi,
        "time_prec_macro": prec_ma, "time_rec_macro": rec_ma, "time_f1_macro": f1_ma,
        "time_bal_acc":    bal_acc,
        "time_acc":        accuracy_from_logits(time_logits, y_time, thr),
        "time_f1_per_class": f1_vec,  # (C,)
    }


def prevalence_per_class(target: torch.Tensor) -> torch.Tensor:
    """Fraction of positives per class in a batch/array (returns (C,))."""
    C = target.shape[1]
    t_flat = target.reshape(target.shape[0], C, -1).transpose(0, 1).reshape(C, -1)
    return t_flat.mean(dim=1)
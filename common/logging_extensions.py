import os, csv, json, math, logging, datetime
from typing import Iterable, Dict, Optional, List, Tuple

# ---------- paths and run structure ----------
def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def make_run_dirs(base_dir: str, run_name: str, annot_name: Optional[str] = None) -> Dict[str, str]:
    """
    Creates: <base>/<annot>/<run>/{logs,checkpoints}
    Returns dict with useful paths.
    """
    parts = [base_dir]
    if annot_name: parts.append(annot_name)
    parts.append(run_name)
    root = os.path.join(*parts)

    logs_dir = os.path.join(root, "logs")
    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    return {
        "root": root,
        "logs_dir": logs_dir,
        "ckpt_dir": ckpt_dir,
        "log_file": os.path.join(logs_dir, "train.log"),
        "csv_file": os.path.join(logs_dir, "metrics.csv"),
        "cfg_file": os.path.join(root, "config.json"),
        "plot_file": os.path.join(logs_dir, "training_curves.html"),
    }

def save_config(cfg: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Base Python logger (console + file) ----------
def setup_logger(log_path: str, name: str = "trainer") -> logging.Logger:
    """
    Console + file logger (INFO level)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt); sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger

# ---------- CSV metrics handler (as a logging.Handler) ----------
class CSVMetricsHandler(logging.Handler):
    """
    Appends selected fields to CSV on logger calls like:
        logger.info("metrics", extra={"metrics": {...}})
    CSV header is fixed on creation; missing keys get "".
    """
    def __init__(self, csv_path: str, header: Iterable[str]):
        super().__init__(level=logging.INFO)
        self.csv_path = csv_path
        self.header = list(header)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.header)

    def emit(self, record: logging.LogRecord):
        row_dict = getattr(record, "metrics", None)
        if row_dict is None:
            return
        row = [row_dict.get(k, "") for k in self.header]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

def attach_csv_handler(logger: logging.Logger, csv_path: str, header: Iterable[str]) -> None:
    logger.addHandler(CSVMetricsHandler(csv_path, header))

def log_metrics(logger: logging.Logger, row: Dict) -> None:
    """
    Emit a metrics row that the CSVMetricsHandler will capture.
    """
    logger.info("metrics", extra={"metrics": row})

# ---------- Checkpoint manager (keep best-K) ----------
class CheckpointManager:
    """
    Keeps top-K checkpoints by a monitored metric (min or max).
    Saves: <ckpt_dir>/<prefix>_epoch{E}_score{S}.pt
    Also always updates: <ckpt_dir>/<prefix>_last.pt
    """
    def __init__(
        self,
        ckpt_dir: str,
        mode: str = "min",                   # "min" for val_loss, "max" for accuracy/F1
        k: int = 3,
        filename_prefix: str = "model",
    ):
        assert mode in ("min", "max")
        self.ckpt_dir = ckpt_dir
        self.mode = mode
        self.k = int(k)
        self.prefix = filename_prefix
        self._best: List[Tuple[float, str, int]] = []  # list of (score, path, epoch)

    def _is_better(self, score: float, best_score: float) -> bool:
        return score < best_score if self.mode == "min" else score > best_score

    def _sort_key(self, item: Tuple[float, str, int]):
        return item[0] if self.mode == "min" else -item[0]

    def step(self, epoch: int, score: float, state_dict: Dict, optimizer_state: Optional[Dict] = None) -> bool:
        """
        Save "last" and maybe keep in top-K. Returns True if it's a top-K improvement.
        """
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # always save 'last'
        last_path = os.path.join(self.ckpt_dir, f"{self.prefix}_last.pt")
        torch_payload = {
            "epoch": epoch,
            "score": score,
            "state_dict": state_dict,
            "optimizer_state": optimizer_state,
        }
        try:
            import torch
            torch.save(torch_payload, last_path)
        except Exception:
            pass

        improved = False
        if len(self._best) < self.k:
            improved = True
        else:
            worst_score = max(self._best, key=lambda x: self._sort_key(x))[0] if self.mode=="min" else \
                          min(self._best, key=lambda x: self._sort_key(x))[0]
            if self._is_better(score, worst_score):
                improved = True

        if improved:
            # save with tagged filename
            safe_score = f"{score:.6f}".replace(".", "_")
            path = os.path.join(self.ckpt_dir, f"{self.prefix}_epoch{epoch:03d}_score{safe_score}.pt")
            try:
                import torch
                torch.save(torch_payload, path)
            except Exception:
                return False

            self._best.append((score, path, epoch))
            # keep only best K (by mode)
            self._best.sort(key=self._sort_key)
            self._best = self._best[: self.k]
            # optionally prune files beyond top-K
            keep_paths = {p for _, p, _ in self._best}
            for fname in os.listdir(self.ckpt_dir):
                if fname.startswith(self.prefix) and "epoch" in fname and fname.endswith(".pt"):
                    full = os.path.join(self.ckpt_dir, fname)
                    if full not in keep_paths:
                        try:
                            os.remove(full)
                        except Exception:
                            print("Not able to prune\n")
                            pass
            return True
        return False
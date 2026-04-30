import datetime
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEY_COLS = ["subject_id", "sleep_date", "lifelog_date"]

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SUB_PATH = BASE_DIR / "ch2026_submission_sample.csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
OOF_DIR = OUTPUTS_DIR / "oof"
REPORT_DIR = OUTPUTS_DIR / "report"
SUMMARY_DIR = OUTPUTS_DIR / "summary"
LOG_DIR = OUTPUTS_DIR / "log"

OUTPUT_PATH = SUBMISSIONS_DIR / "submission_test_deepstack.csv"
OOF_PATH = OOF_DIR / "oof_test_deepstack.csv"
TEST_PREDS_PATH = REPORT_DIR / "test_preds_test_deepstack.csv"
REPORT_PATH = REPORT_DIR / "report_test_deepstack.txt"
SUMMARY_PATH = SUMMARY_DIR / "summary_test_deepstack.json"
RUN_LOG_PATH = LOG_DIR / "run_test_deepstack.log"

CLIP_LO = float(os.environ.get("TEST_CLIP_LO", "0.02"))
CLIP_HI = float(os.environ.get("TEST_CLIP_HI", "0.98"))
PSEUDO_PUBLIC_TAIL_FRAC = float(os.environ.get("TEST_PSEUDO_TAIL_FRAC", "0.2"))

SEEDS = [int(x) for x in os.environ.get("TEST_SEEDS", "42,2025").split(",") if x.strip()]
N_FOLDS = int(os.environ.get("TEST_FOLDS", "5"))
BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "128"))
EPOCHS = int(os.environ.get("TEST_EPOCHS", "120"))
PATIENCE = int(os.environ.get("TEST_PATIENCE", "20"))
LR = float(os.environ.get("TEST_LR", "0.002"))
WEIGHT_DECAY = float(os.environ.get("TEST_WEIGHT_DECAY", "0.0001"))
HIDDEN1 = int(os.environ.get("TEST_HIDDEN1", "256"))
HIDDEN2 = int(os.environ.get("TEST_HIDDEN2", "128"))
DROPOUT = float(os.environ.get("TEST_DROPOUT", "0.25"))
USE_AMP = os.environ.get("TEST_USE_AMP", "1") == "1"
FORCE_CPU = os.environ.get("TEST_FORCE_CPU", "0") == "1"
DRY_RUN = os.environ.get("TEST_DRY_RUN", "0") == "1"

DEFAULT_MODEL_SPECS = [
    (
        "A",
        "outputs/oof/oof_v12_public_v12_foldsafe_te.csv",
        "outputs/submissions/submission_v12_public_v12_foldsafe_te.csv",
    ),
    (
        "B",
        "outputs/oof/oof_v10_public_base_trainnorm_rankblend.csv",
        "outputs/report/test_preds_v10_public_base_trainnorm_rankblend.csv",
    ),
    (
        "C",
        "outputs/oof/oof_v14.csv",
        "outputs/submissions/submission_v14.csv",
    ),
    (
        "D",
        "outputs/oof/oof_v9.csv",
        "outputs/report/test_preds_v9.csv",
    ),
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class DeepStackMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def ensure_dirs() -> None:
    for d in [SUBMISSIONS_DIR, OOF_DIR, REPORT_DIR, SUMMARY_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def parse_model_specs() -> List[Tuple[str, str, str]]:
    raw = os.environ.get("TEST_MODEL_SPECS", "").strip()
    if not raw:
        return DEFAULT_MODEL_SPECS
    specs = []
    # format: alias|oof_path|test_path;alias|oof_path|test_path
    for block in raw.split(";"):
        blk = block.strip()
        if not blk:
            continue
        parts = [x.strip() for x in blk.split("|")]
        if len(parts) != 3:
            raise ValueError(f"Invalid TEST_MODEL_SPECS block: {blk}")
        specs.append((parts[0], parts[1], parts[2]))
    if not specs:
        raise ValueError("TEST_MODEL_SPECS resolved to empty list")
    return specs


def resolve_path(spec: str) -> Path:
    p = Path(spec)
    if p.is_absolute():
        return p
    return BASE_DIR / spec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_pseudo_public_mask(df: pd.DataFrame, tail_frac: float) -> np.ndarray:
    mask = pd.Series(False, index=df.index)
    ordered = df.sort_values(["subject_id", "lifelog_date"])
    for _, grp in ordered.groupby("subject_id"):
        n = len(grp)
        tail_n = max(1, int(np.ceil(n * tail_frac)))
        mask.loc[grp.index[-tail_n:]] = True
    return mask.values


def load_oof_matrix(path: Path, train_df: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path)
    if all(f"oof_{t}" in df.columns for t in TARGETS):
        if len(df) != len(train_df):
            raise ValueError(f"{path}: length mismatch for oof_ format")
        return np.column_stack([df[f"oof_{t}"].astype(float).values for t in TARGETS])
    if all(f"pred_{t}" in df.columns for t in TARGETS):
        if not {"subject_id", "lifelog_date"}.issubset(df.columns):
            raise ValueError(f"{path}: pred_ format requires subject_id/lifelog_date")
        z = df.copy()
        left = train_df[["subject_id", "lifelog_date"]].copy()
        left["subject_id"] = left["subject_id"].astype(str)
        left["lifelog_date"] = left["lifelog_date"].astype(str)
        z["subject_id"] = z["subject_id"].astype(str)
        z["lifelog_date"] = z["lifelog_date"].astype(str)
        m = left.merge(
            z[["subject_id", "lifelog_date"] + [f"pred_{t}" for t in TARGETS]],
            on=["subject_id", "lifelog_date"],
            how="left",
            validate="one_to_one",
        )
        if m[[f"pred_{t}" for t in TARGETS]].isna().any().any():
            raise ValueError(f"{path}: missing rows after merge")
        return np.column_stack([m[f"pred_{t}"].astype(float).values for t in TARGETS])
    if all(t in df.columns for t in TARGETS):
        if {"subject_id", "lifelog_date"}.issubset(df.columns):
            z = df.copy()
            left = train_df[["subject_id", "lifelog_date"]].copy()
            left["subject_id"] = left["subject_id"].astype(str)
            left["lifelog_date"] = left["lifelog_date"].astype(str)
            z["subject_id"] = z["subject_id"].astype(str)
            z["lifelog_date"] = z["lifelog_date"].astype(str)
            m = left.merge(
                z[["subject_id", "lifelog_date"] + TARGETS],
                on=["subject_id", "lifelog_date"],
                how="left",
                validate="one_to_one",
            )
            if m[TARGETS].isna().any().any():
                raise ValueError(f"{path}: missing rows after merge")
            return np.column_stack([m[t].astype(float).values for t in TARGETS])
        if len(df) == len(train_df):
            return np.column_stack([df[t].astype(float).values for t in TARGETS])
    raise ValueError(f"Unsupported OOF format: {path}")


def load_test_matrix(path: Path, sample_df: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path)
    if all(t in df.columns for t in TARGETS):
        if set(KEY_COLS).issubset(df.columns):
            z = df.copy()
            left = sample_df[KEY_COLS].copy()
            for c in KEY_COLS:
                left[c] = left[c].astype(str)
                z[c] = z[c].astype(str)
            m = left.merge(
                z[KEY_COLS + TARGETS],
                on=KEY_COLS,
                how="left",
                validate="one_to_one",
            )
            if m[TARGETS].isna().any().any():
                raise ValueError(f"{path}: missing rows after merge")
            return np.column_stack([m[t].astype(float).values for t in TARGETS])
        if len(df) == len(sample_df):
            return np.column_stack([df[t].astype(float).values for t in TARGETS])
    if all(f"pred_{t}" in df.columns for t in TARGETS):
        if not set(KEY_COLS).issubset(df.columns):
            raise ValueError(f"{path}: pred_ format requires KEY_COLS")
        z = df.copy()
        left = sample_df[KEY_COLS].copy()
        for c in KEY_COLS:
            left[c] = left[c].astype(str)
            z[c] = z[c].astype(str)
        m = left.merge(
            z[KEY_COLS + [f"pred_{t}" for t in TARGETS]],
            on=KEY_COLS,
            how="left",
            validate="one_to_one",
        )
        if m[[f"pred_{t}" for t in TARGETS]].isna().any().any():
            raise ValueError(f"{path}: missing rows after merge")
        return np.column_stack([m[f"pred_{t}"].astype(float).values for t in TARGETS])
    raise ValueError(f"Unsupported test prediction format: {path}")


def build_target_features(
    pred_map: Dict[str, np.ndarray], aliases: List[str], target_idx: int
) -> np.ndarray:
    chunks = []
    stack = []
    for alias in aliases:
        arr = pred_map[alias]
        chunks.append(arr)
        chunks.append(safe_logit(arr[:, target_idx])[:, None])
        stack.append(arr[:, target_idx])
    stack = np.column_stack(stack)
    chunks.extend(
        [
            stack.mean(axis=1, keepdims=True),
            stack.std(axis=1, keepdims=True),
            stack.min(axis=1, keepdims=True),
            stack.max(axis=1, keepdims=True),
        ]
    )
    X = np.concatenate(chunks, axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def predict_prob(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(xt)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


def train_single_target(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    target_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    n_test = len(X_test)
    oof_all = np.zeros(n, dtype=float)
    test_all = np.zeros(n_test, dtype=float)

    use_amp = (device.type == "cuda") and USE_AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dtype_amp = torch.float16

    for seed in SEEDS:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        seed_oof = np.zeros(n, dtype=float)
        seed_test = np.zeros(n_test, dtype=float)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            x_tr = X[tr_idx].copy()
            x_val = X[val_idx].copy()
            x_te = X_test.copy()
            y_tr = y[tr_idx].astype(np.float32)
            y_val = y[val_idx].astype(np.float32)

            mu = x_tr.mean(axis=0, keepdims=True)
            sigma = x_tr.std(axis=0, keepdims=True)
            sigma[sigma < 1e-6] = 1.0
            x_tr = (x_tr - mu) / sigma
            x_val = (x_val - mu) / sigma
            x_te = (x_te - mu) / sigma

            x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0)
            x_val = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_te = np.nan_to_num(x_te, nan=0.0, posinf=0.0, neginf=0.0)

            train_ds = TensorDataset(
                torch.tensor(x_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=False,
            )

            model = DeepStackMLP(in_dim=x_tr.shape[1]).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )

            best_val_loss = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(EPOCHS):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type="cuda",
                        dtype=dtype_amp,
                        enabled=use_amp,
                    ):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                model.eval()
                with torch.no_grad():
                    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
                    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
                    val_logits = model(xv)
                    val_loss = criterion(val_logits, yv).item()

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        break

            if best_state is None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)

            val_pred = predict_prob(model, x_val, device)
            test_pred = predict_prob(model, x_te, device)
            seed_oof[val_idx] = val_pred
            seed_test += test_pred / N_FOLDS

            fold_ll = log_loss(y_val, np.clip(val_pred, 1e-6, 1 - 1e-6))
            print(
                f"  {target_name} | seed={seed} fold={fold}/{N_FOLDS} "
                f"val_logloss={fold_ll:.5f}"
            )

        seed_ll = log_loss(y, np.clip(seed_oof, 1e-6, 1 - 1e-6))
        print(f"  {target_name} | seed={seed} OOF={seed_ll:.5f}")
        oof_all += seed_oof
        test_all += seed_test

    oof_all /= len(SEEDS)
    test_all /= len(SEEDS)
    oof_all = np.clip(oof_all, 1e-6, 1 - 1e-6)
    test_all = np.clip(test_all, 1e-6, 1 - 1e-6)
    return oof_all, test_all


def main() -> None:
    ensure_dirs()

    _stdout, _stderr = os.sys.stdout, os.sys.stderr
    run_log = open(RUN_LOG_PATH, "w", encoding="utf-8")
    os.sys.stdout = Tee(_stdout, run_log)
    os.sys.stderr = Tee(_stderr, run_log)

    try:
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not FORCE_CPU
            else torch.device("cpu")
        )
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"CUDA: {torch.cuda.get_device_name(0)}")

        if CLIP_LO >= CLIP_HI:
            raise ValueError("TEST_CLIP_LO must be < TEST_CLIP_HI")
        if not (0 < PSEUDO_PUBLIC_TAIL_FRAC < 1):
            raise ValueError("TEST_PSEUDO_TAIL_FRAC must be in (0, 1)")

        print("Loading train/sample...")
        train_df = pd.read_csv(TRAIN_PATH)
        sample_df = pd.read_csv(SUB_PATH)
        train_df["lifelog_date"] = pd.to_datetime(train_df["lifelog_date"])
        sample_df["lifelog_date"] = pd.to_datetime(sample_df["lifelog_date"])

        model_specs = parse_model_specs()
        aliases = [alias for alias, _, _ in model_specs]
        print(f"Using model aliases: {aliases}")

        train_pred_map: Dict[str, np.ndarray] = {}
        test_pred_map: Dict[str, np.ndarray] = {}

        for alias, oof_spec, test_spec in model_specs:
            oof_path = resolve_path(oof_spec)
            test_path = resolve_path(test_spec)
            print(f"[{alias}] oof={oof_path}")
            print(f"[{alias}] test={test_path}")
            train_pred_map[alias] = load_oof_matrix(oof_path, train_df)
            test_pred_map[alias] = load_test_matrix(test_path, sample_df)

        n_train = len(train_df)
        n_test = len(sample_df)
        for alias in aliases:
            if train_pred_map[alias].shape != (n_train, len(TARGETS)):
                raise ValueError(f"{alias}: invalid train matrix shape {train_pred_map[alias].shape}")
            if test_pred_map[alias].shape != (n_test, len(TARGETS)):
                raise ValueError(f"{alias}: invalid test matrix shape {test_pred_map[alias].shape}")

        oof_preds = np.zeros((n_train, len(TARGETS)), dtype=float)
        test_preds = np.zeros((n_test, len(TARGETS)), dtype=float)

        if DRY_RUN:
            print("DRY_RUN=1 -> skip training, use mean blend of aliases.")
            for ti in range(len(TARGETS)):
                oof_preds[:, ti] = np.mean(
                    [train_pred_map[a][:, ti] for a in aliases], axis=0
                )
                test_preds[:, ti] = np.mean(
                    [test_pred_map[a][:, ti] for a in aliases], axis=0
                )
        else:
            for ti, target in enumerate(TARGETS):
                y = train_df[target].astype(int).values
                X = build_target_features(train_pred_map, aliases, ti)
                X_test = build_target_features(test_pred_map, aliases, ti)
                print(
                    f"\n=== Target {target} | pos_rate={y.mean():.3f} | "
                    f"features={X.shape[1]} ==="
                )
                oof_t, test_t = train_single_target(X, y, X_test, device, target)
                oof_preds[:, ti] = oof_t
                test_preds[:, ti] = test_t
                print(f"  {target} final OOF: {log_loss(y, oof_t):.5f}")

        per_target_oof = {
            t: float(log_loss(train_df[t].values, oof_preds[:, i]))
            for i, t in enumerate(TARGETS)
        }
        avg_oof = float(np.mean(list(per_target_oof.values())))

        pseudo_mask = build_pseudo_public_mask(
            train_df[["subject_id", "lifelog_date"]], PSEUDO_PUBLIC_TAIL_FRAC
        )
        pseudo_per_target = {
            t: float(log_loss(train_df.loc[pseudo_mask, t].values, oof_preds[pseudo_mask, i]))
            for i, t in enumerate(TARGETS)
        }
        pseudo_oof = float(np.mean(list(pseudo_per_target.values())))

        print("\n" + "=" * 72)
        print(f"test.py DeepStack avg OOF:         {avg_oof:.6f}")
        print(f"test.py DeepStack pseudo-public:   {pseudo_oof:.6f}")
        print("=" * 72)

        oof_out = train_df[KEY_COLS + TARGETS].copy()
        for i, t in enumerate(TARGETS):
            oof_out[f"pred_{t}"] = oof_preds[:, i]
        oof_out.to_csv(OOF_PATH, index=False)

        test_out = sample_df[KEY_COLS].copy()
        for i, t in enumerate(TARGETS):
            test_out[t] = test_preds[:, i]
        test_out.to_csv(TEST_PREDS_PATH, index=False)

        submission = sample_df[KEY_COLS].copy()
        for i, t in enumerate(TARGETS):
            submission[t] = np.clip(test_preds[:, i], CLIP_LO, CLIP_HI)
        submission.to_csv(OUTPUT_PATH, index=False)

        report_lines = [
            "=" * 80,
            "test.py deep stacking report",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Device: {device}",
            f"Seeds: {SEEDS}",
            f"Folds: {N_FOLDS}",
            f"Epochs/Patience: {EPOCHS}/{PATIENCE}",
            f"Batch size: {BATCH_SIZE}",
            f"LR/WD: {LR}/{WEIGHT_DECAY}",
            f"Hidden: {HIDDEN1}, {HIDDEN2} | dropout={DROPOUT}",
            f"Use AMP: {USE_AMP and device.type == 'cuda'}",
            f"Pseudo tail frac: {PSEUDO_PUBLIC_TAIL_FRAC}",
            "",
            "[Model specs]",
        ]
        for alias, oof_spec, test_spec in model_specs:
            report_lines.append(f"  {alias}: oof={oof_spec} | test={test_spec}")
        report_lines += [
            "",
            f"[Summary] avg_oof={avg_oof:.6f}, pseudo_oof={pseudo_oof:.6f}",
            "[Per target OOF]",
        ]
        for t in TARGETS:
            report_lines.append(f"  {t}: {per_target_oof[t]:.6f}")
        report_text = "\n".join(report_lines)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(report_text)

        summary = {
            "exp_tag": "test_deepstack",
            "device": str(device),
            "seeds": SEEDS,
            "n_folds": N_FOLDS,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden1": HIDDEN1,
            "hidden2": HIDDEN2,
            "dropout": DROPOUT,
            "use_amp": bool(USE_AMP and device.type == "cuda"),
            "pseudo_tail_frac": PSEUDO_PUBLIC_TAIL_FRAC,
            "model_specs": [
                {"alias": a, "oof": o, "test": t} for a, o, t in model_specs
            ],
            "avg_oof": avg_oof,
            "pseudo_public_oof": pseudo_oof,
            "per_target_oof": per_target_oof,
            "pseudo_public_per_target_oof": pseudo_per_target,
            "artifacts": {
                "submission": str(OUTPUT_PATH),
                "oof": str(OOF_PATH),
                "test_preds": str(TEST_PREDS_PATH),
                "report": str(REPORT_PATH),
                "summary": str(SUMMARY_PATH),
                "run_log": str(RUN_LOG_PATH),
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[saved] {OUTPUT_PATH}")
        print(f"[saved] {SUMMARY_PATH}")
    finally:
        os.sys.stdout = _stdout
        os.sys.stderr = _stderr
        run_log.close()


if __name__ == "__main__":
    main()

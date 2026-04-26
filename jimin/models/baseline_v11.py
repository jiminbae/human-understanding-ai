import json
import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEY_COLS = ["subject_id", "sleep_date", "lifelog_date"]

BASE_DIR = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = BASE_DIR / "outputs" / "submissions"
REPORT_DIR = BASE_DIR / "outputs" / "report"
SUMMARY_DIR = BASE_DIR / "outputs" / "summary"

# v11 default: anchor the best known public file, blend with a diverse file in a small ratio.
BASE_SUB_NAME = os.environ.get(
    "V11_BASE_SUB",
    "submission_v10_public_stable_reproduce.csv",
).strip()
BLEND_SUB_NAME = os.environ.get(
    "V11_BLEND_SUB", "submission_v9_.csv").strip()

# Final = alpha * base + (1 - alpha) * blend
ALPHA = float(os.environ.get("V11_ALPHA", "0.95"))
CLIP_LO = float(os.environ.get("V11_CLIP_LO", "0.02"))
CLIP_HI = float(os.environ.get("V11_CLIP_HI", "0.98"))


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def read_submission_csv(name: str) -> pd.DataFrame:
    path = SUBMISSIONS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in KEY_COLS + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")

    return df


def validate_alignment(df_base: pd.DataFrame, df_blend: pd.DataFrame) -> None:
    if len(df_base) != len(df_blend):
        raise ValueError(
            f"Row count mismatch: base={len(df_base)}, blend={len(df_blend)}"
        )

    left = df_base[KEY_COLS].copy()
    right = df_blend[KEY_COLS].copy()

    for c in ["sleep_date", "lifelog_date"]:
        left[c] = left[c].astype(str)
        right[c] = right[c].astype(str)

    mismatch = ~(left.values == right.values).all(axis=1)
    if mismatch.any():
        idx = int(np.where(mismatch)[0][0])
        raise ValueError(
            "Key columns are not aligned between submissions. "
            f"First mismatch at row {idx}: "
            f"base={left.iloc[idx].to_dict()}, blend={right.iloc[idx].to_dict()}"
        )


def blend_submissions(df_base: pd.DataFrame, df_blend: pd.DataFrame) -> pd.DataFrame:
    if not (0.0 <= ALPHA <= 1.0):
        raise ValueError(f"V11_ALPHA must be between 0 and 1, got {ALPHA}")
    if CLIP_LO >= CLIP_HI:
        raise ValueError(
            f"Invalid clip range: CLIP_LO={CLIP_LO}, CLIP_HI={CLIP_HI}"
        )

    out = df_base[KEY_COLS].copy()
    for t in TARGETS:
        v = ALPHA * df_base[t].astype(float).values + (1.0 - ALPHA) * df_blend[t].astype(float).values
        out[t] = np.clip(v, CLIP_LO, CLIP_HI)
    return out


def run() -> None:
    ensure_dirs()

    df_base = read_submission_csv(BASE_SUB_NAME)
    df_blend = read_submission_csv(BLEND_SUB_NAME)
    validate_alignment(df_base, df_blend)

    submission = blend_submissions(df_base, df_blend)

    base_tag = Path(BASE_SUB_NAME).stem.replace("submission_", "")
    blend_tag = Path(BLEND_SUB_NAME).stem.replace("submission_", "")
    alpha_tag = str(ALPHA).replace(".", "p")
    exp_tag = f"_blend_{base_tag}__{blend_tag}__a{alpha_tag}"

    submission_path = SUBMISSIONS_DIR / f"submission_v11{exp_tag}.csv"
    report_path = REPORT_DIR / f"report_v11{exp_tag}.txt"
    summary_path = SUMMARY_DIR / f"summary_v11{exp_tag}.json"

    submission.to_csv(submission_path, index=False)

    lines = []
    lines.append("=" * 80)
    lines.append("Baseline v11 blend report")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("[Blend setup]")
    lines.append(f"  Base submission : {BASE_SUB_NAME}")
    lines.append(f"  Blend submission: {BLEND_SUB_NAME}")
    lines.append(f"  Alpha(base)     : {ALPHA:.6f}")
    lines.append(f"  Alpha(blend)    : {1.0 - ALPHA:.6f}")
    lines.append(f"  Clip range      : [{CLIP_LO}, {CLIP_HI}]")
    lines.append(f"  Rows            : {len(submission)}")
    lines.append("")
    lines.append("[Prediction stats]")
    for t in TARGETS:
        vals = submission[t].astype(float).values
        lines.append(
            f"  {t}: mean={vals.mean():.6f}, std={vals.std():.6f}, min={vals.min():.6f}, max={vals.max():.6f}"
        )
    lines.append("")
    lines.append(f"[Output] {submission_path}")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    summary = {
        "base_submission": BASE_SUB_NAME,
        "blend_submission": BLEND_SUB_NAME,
        "alpha_base": ALPHA,
        "alpha_blend": 1.0 - ALPHA,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "n_rows": int(len(submission)),
        "exp_tag": exp_tag,
        "artifacts": {
            "submission": str(submission_path),
            "report": str(report_path),
            "summary": str(summary_path),
        },
        "prediction_stats": {
            t: {
                "mean": float(submission[t].mean()),
                "std": float(submission[t].std()),
                "min": float(submission[t].min()),
                "max": float(submission[t].max()),
            }
            for t in TARGETS
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(report_text)
    print(f"[report saved] {report_path}")
    print(f"[summary saved] {summary_path}")


if __name__ == "__main__":
    run()

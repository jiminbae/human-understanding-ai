# v55: follow-up around the new public best v54_public_mid_no_q3s3.
#
# Public feedback now says:
#   - public_mid improved over v48 base.
#   - s4_half got worse, so S4 delta should stay.
#   - no_q3s3 improved, so Q3/S3 deltas were harmful.
#
# The next useful questions are:
#   1. Is Q2 helpful, harmful, or neutral?
#   2. After removing Q3/S3, can the remaining useful targets be pushed a bit?
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

BASE_TAG = 'v48_target_delta_scaled_avg430_q2cap115_q3s3guard'
MID_TAG = 'v53_public_mid_probe'
BEST_TAG = 'v54_public_mid_no_q3s3'


def load_pair(tag):
    return (
        pd.read_csv(OOF_DIR / f'oof_{tag}.csv'),
        pd.read_csv(SUB_DIR / f'submission_{tag}.csv'),
    )


def clip_frame(frame):
    out = frame.copy()
    out[TARGETS] = out[TARGETS].clip(0.02, 0.98)
    return out


def weighted_from_base(base, mid, weights):
    out = base[KEYS].copy()
    for target in TARGETS:
        out[target] = base[target] + float(weights[target]) * (mid[target] - base[target])
    return clip_frame(out)


def target_logloss(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 0.02, 0.98)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate(train, pred):
    per = {target: target_logloss(train[target], pred[target]) for target in TARGETS}
    return float(np.mean(list(per.values()))), per


def describe(pred, base):
    diff = pred[TARGETS] - base[TARGETS]
    return {
        'mad_vs_base': float(diff.abs().to_numpy().mean()),
        'max_abs_vs_base': float(diff.abs().to_numpy().max()),
        'per_target_mad': {target: float(diff[target].abs().mean()) for target in TARGETS},
        'mean_delta': {target: float(diff[target].mean()) for target in TARGETS},
        'means': {target: float(pred[target].mean()) for target in TARGETS},
    }


def save_candidate(name, train, base_oof, mid_oof, base_sub, mid_sub, weights, note):
    oof = weighted_from_base(base_oof, mid_oof, weights)
    sub = weighted_from_base(base_sub, mid_sub, weights)
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    sub.to_csv(sub_path, index=False)
    loss, per = evaluate(train, oof)
    return {
        'name': name,
        'weights': weights,
        'note': note,
        'oof_proxy_loss': loss,
        'oof_proxy_per_target': per,
        'distribution_vs_base': describe(sub, base_sub),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
    }


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    base_oof, base_sub = load_pair(BASE_TAG)
    mid_oof, mid_sub = load_pair(MID_TAG)

    candidates = []
    specs = [
        (
            'v55_no_q3s3_q2_half',
            {'Q1': 1.0, 'Q2': 0.5, 'Q3': 0.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.0},
            'Q2 half ablation: if Q2 is mildly harmful this should improve over no_q3s3.',
        ),
        (
            'v55_no_q3s3_no_q2',
            {'Q1': 1.0, 'Q2': 0.0, 'Q3': 0.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.0},
            'Q2 full ablation; same surface as v54 core_q1s1s2_s4 but with explicit v55 name.',
        ),
        (
            'v55_no_q3s3_scale104',
            {'Q1': 1.04, 'Q2': 1.04, 'Q3': 0.0, 'S1': 1.04, 'S2': 1.04, 'S3': 0.0, 'S4': 1.04},
            'Small scale-up after removing harmful Q3/S3.',
        ),
        (
            'v55_no_q3s3_scale108',
            {'Q1': 1.08, 'Q2': 1.08, 'Q3': 0.0, 'S1': 1.08, 'S2': 1.08, 'S3': 0.0, 'S4': 1.08},
            'Moderate scale-up after removing harmful Q3/S3.',
        ),
        (
            'v55_no_q3s3_core_scale108_q2_keep',
            {'Q1': 1.08, 'Q2': 1.0, 'Q3': 0.0, 'S1': 1.08, 'S2': 1.08, 'S3': 0.0, 'S4': 1.08},
            'Scale strong targets but keep Q2 at validated no_q3s3 weight.',
        ),
        (
            'v55_no_q3s3_s4_boost115',
            {'Q1': 1.0, 'Q2': 1.0, 'Q3': 0.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.15},
            'Only boost S4; s4_half worsened, so S4 may have room above 1.0.',
        ),
    ]
    for name, weights, note in specs:
        candidates.append(save_candidate(name, train, base_oof, mid_oof, base_sub, mid_sub, weights, note))

    summary = {
        'exp_tag': 'v55_no_q3s3_followup',
        'known_public_scores': {
            BASE_TAG: 0.5805824813,
            MID_TAG: 0.5800163708,
            'v54_public_mid_s4_half': 0.5800565394,
            BEST_TAG: 0.5799096236,
        },
        'recommended_next_submit_order': [
            'v55_no_q3s3_q2_half',
            'v55_no_q3s3_core_scale108_q2_keep',
            'v55_no_q3s3_s4_boost115',
        ],
        'candidates': sorted(candidates, key=lambda x: x['oof_proxy_loss']),
    }
    path = SUMMARY_DIR / 'summary_v55_no_q3s3_followup.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[v55] summary={path}')
    for item in summary['candidates']:
        print(
            f"  {item['name']}: proxy={item['oof_proxy_loss']:.6f} "
            f"mad={item['distribution_vs_base']['mad_vs_base']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

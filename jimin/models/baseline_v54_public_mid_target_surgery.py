# v54: target-wise surgery around the public-confirmed v53_public_mid_probe.
#
# Public feedback:
#   - v53_public_mid_probe improved over the v48 base.
#   - scaling the same direction to 1.15 got worse.
#
# That means the direction is useful, but at least one target/role is already
# over-pushed.  This script creates targeted ablations so the next two public
# submissions can identify whether S4 or the weak small-delta targets are the
# limiting part.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

BASE_TAG = 'v48_target_delta_scaled_avg430_q2cap115_q3s3guard'
MID_TAG = 'v53_public_mid_probe'


def load_pair(tag):
    oof = pd.read_csv(OOF_DIR / f'oof_{tag}.csv')
    sub = pd.read_csv(SUB_DIR / f'submission_{tag}.csv')
    return oof, sub


def clip_frame(frame):
    out = frame.copy()
    out[TARGETS] = out[TARGETS].clip(0.02, 0.98)
    return out


def target_weights_blend(base, mid, weights):
    out = base[KEYS].copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        out[target] = base[target] + weight * (mid[target] - base[target])
    return clip_frame(out)


def target_logloss(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 0.02, 0.98)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate(train, pred):
    per_target = {
        target: target_logloss(train[target], pred[target])
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def describe_vs_base(pred, base):
    diff = pred[TARGETS] - base[TARGETS]
    return {
        'mad_vs_base': float(diff.abs().to_numpy().mean()),
        'max_abs_vs_base': float(diff.abs().to_numpy().max()),
        'per_target_mad': {
            target: float(diff[target].abs().mean())
            for target in TARGETS
        },
        'mean_delta': {
            target: float(diff[target].mean())
            for target in TARGETS
        },
        'means': {
            target: float(pred[target].mean())
            for target in TARGETS
        },
    }


def save_candidate(name, train, base_oof, mid_oof, base_sub, mid_sub, weights, policy_note):
    oof = target_weights_blend(base_oof, mid_oof, weights)
    sub = target_weights_blend(base_sub, mid_sub, weights)
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    sub.to_csv(sub_path, index=False)
    loss, per_target = evaluate(train, oof)
    return {
        'name': name,
        'weights': weights,
        'policy_note': policy_note,
        'oof_proxy_loss': loss,
        'oof_proxy_per_target': per_target,
        'distribution_vs_base': describe_vs_base(sub, base_sub),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
    }


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    base_oof, base_sub = load_pair(BASE_TAG)
    mid_oof, mid_sub = load_pair(MID_TAG)

    specs = [
        (
            'v54_public_mid_s4_half',
            {'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 1.0, 'S2': 1.0, 'S3': 1.0, 'S4': 0.5},
            'First-submit branch test: keep public_mid except halve S4 delta.',
        ),
        (
            'v54_public_mid_no_s4',
            {'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 1.0, 'S2': 1.0, 'S3': 1.0, 'S4': 0.0},
            'Use as second submit if s4_half improves; tests whether S4 delta is harmful.',
        ),
        (
            'v54_public_mid_core_q1s1s2',
            {'Q1': 1.0, 'Q2': 0.0, 'Q3': 0.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 0.0},
            'Core sensor axis only; tests whether non-core target deltas are noise.',
        ),
        (
            'v54_public_mid_core_q1s1s2_s4',
            {'Q1': 1.0, 'Q2': 0.0, 'Q3': 0.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.0},
            'More aggressive fallback; keeps S4 but drops weak Q2/Q3/S3 deltas.',
        ),
        (
            'v54_public_mid_no_q3s3',
            {'Q1': 1.0, 'Q2': 1.0, 'Q3': 0.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.0},
            'Keeps Q2/S4 but removes the tiny Q3/S3 deltas.',
        ),
    ]
    candidates = [
        save_candidate(name, train, base_oof, mid_oof, base_sub, mid_sub, weights, note)
        for name, weights, note in specs
    ]
    summary = {
        'exp_tag': 'v54_public_mid_target_surgery',
        'inputs': {
            'base_tag': BASE_TAG,
            'mid_tag': MID_TAG,
            'known_public_scores': {
                BASE_TAG: 0.5805824813,
                MID_TAG: 0.5800163708,
                'v53_public_mid_scale115': 0.5800428218,
            },
        },
        'recommended_two_submit_plan': [
            'Submit v54_public_mid_s4_half first.',
            'If it improves over v53_public_mid_probe, submit v54_public_mid_no_s4.',
            'If it worsens, submit v54_public_mid_no_q3s3.',
        ],
        'candidates': sorted(candidates, key=lambda row: row['oof_proxy_loss']),
    }
    path = SUMMARY_DIR / 'summary_v54_public_mid_target_surgery.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v54] summary={path}')
    for item in summary['candidates']:
        print(
            f"  {item['name']}: proxy={item['oof_proxy_loss']:.6f} "
            f"mad={item['distribution_vs_base']['mad_vs_base']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

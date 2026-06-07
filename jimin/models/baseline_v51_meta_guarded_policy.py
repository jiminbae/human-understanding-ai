# v51: guarded policy layer for v50 sequence meta predictions.
#
# v50 meta raw has excellent proxy OOF but moves far away from the v48 public
# base. v51 does not train a new model; it asks a narrower question:
# "Can we use v50's routing signal in a public-friendlier, capped way?"
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

META_OOF = OOF_DIR / 'oof_v50_sequence_meta_raw.csv'
META_SUB = SUB_DIR / 'submission_v50_sequence_meta_raw.csv'
RAW_OOF = OOF_DIR / 'oof_v47_hourgrid_subject_state_residual_raw.csv'
RAW_SUB = SUB_DIR / 'submission_v47_hourgrid_subject_state_residual_raw.csv'

DEFAULT_BASE_TAGS = [
    'v48_target_delta_scaled_avg310_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg270_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg250_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg230_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg190_q2cap115_q3s3guard',
]


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['sleep_date', 'lifelog_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.reset_index(drop=True)


def choose_base_prediction():
    env_tag = os.environ.get('V51_BASE_TAG')
    tags = [env_tag] if env_tag else []
    tags.extend(DEFAULT_BASE_TAGS)
    for tag in tags:
        if not tag:
            continue
        oof_path = OOF_DIR / f'oof_{tag}.csv'
        sub_path = SUB_DIR / f'submission_{tag}.csv'
        if oof_path.exists() and sub_path.exists():
            return tag, load_frame(oof_path), load_frame(sub_path), str(oof_path), str(sub_path)
    raise FileNotFoundError('No usable v48 base prediction found.')


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def target_logloss(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(y_pred)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate_frame(train, pred):
    per_target = {
        target: target_logloss(train[target], pred[target])
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def make_keys(df):
    return df[KEYS].copy()


def blend_target_weights(base, meta, weights):
    out = base.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        if weight <= 0:
            continue
        out[target] = clip_prob((1.0 - weight) * base[target] + weight * meta[target])
    return out


def disagreement_guard(base, meta, high_weights, thresholds, low_scale=0.25):
    out = base.copy()
    diagnostics = {}
    for target in TARGETS:
        high = float(high_weights.get(target, 0.0))
        if high <= 0:
            continue
        delta = (meta[target] - base[target]).to_numpy(dtype=float)
        threshold = float(thresholds[target])
        row_weight = np.where(np.abs(delta) <= threshold, high, high * low_scale)
        out[target] = clip_prob((1.0 - row_weight) * base[target] + row_weight * meta[target])
        diagnostics[target] = {
            'threshold': threshold,
            'mean_weight': float(row_weight.mean()),
            'high_rate': float(np.mean(row_weight == high)),
        }
    return out, diagnostics


def raw_meta_agreement_guard(base, meta, raw, high_weights, thresholds, low_weight=0.0, disagree_scale=0.20):
    out = base.copy()
    diagnostics = {}
    for target in TARGETS:
        high = float(high_weights.get(target, 0.0))
        if high <= 0:
            continue
        meta_delta = (meta[target] - base[target]).to_numpy(dtype=float)
        raw_delta = (raw[target] - base[target]).to_numpy(dtype=float)
        same_direction = meta_delta * raw_delta > 0
        small_enough = np.abs(meta_delta) <= float(thresholds[target])
        row_weight = np.full(len(base), low_weight, dtype=float)
        row_weight[same_direction & small_enough] = high
        row_weight[same_direction & (~small_enough)] = high * disagree_scale
        out[target] = clip_prob((1.0 - row_weight) * base[target] + row_weight * meta[target])
        diagnostics[target] = {
            'threshold': float(thresholds[target]),
            'same_direction_rate': float(np.mean(same_direction)),
            'high_rate': float(np.mean(row_weight == high)),
            'mean_weight': float(row_weight.mean()),
        }
    return out, diagnostics


def describe_vs_base(pred, base):
    pred_arr = pred[TARGETS].to_numpy(dtype=float)
    base_arr = base[TARGETS].to_numpy(dtype=float)
    diff = pred_arr - base_arr
    return {
        'corr_vs_base': float(np.corrcoef(pred_arr.ravel(), base_arr.ravel())[0, 1]),
        'mad_vs_base': float(np.mean(np.abs(diff))),
        'max_abs_vs_base': float(np.max(np.abs(diff))),
        'per_target_mad': {
            target: float(np.mean(np.abs(pred[target] - base[target])))
            for target in TARGETS
        },
        'means': {target: float(pred[target].mean()) for target in TARGETS},
    }


def save_candidate(name, train, sub, base_sub, oof_pred, sub_pred, policy):
    oof = make_keys(train)
    submission = make_keys(sub)
    for target in TARGETS:
        oof[target] = clip_prob(oof_pred[target])
        submission[target] = clip_prob(sub_pred[target])
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    oof_loss, per_target = evaluate_frame(train, oof)
    return {
        'name': name,
        'policy': policy,
        'oof_loss': oof_loss,
        'oof_per_target': per_target,
        'distribution_vs_base': describe_vs_base(submission, base_sub),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
    }


def main():
    ensure_dirs()
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_SAMPLE_PATH)
    base_tag, base_oof, base_sub, base_oof_path, base_sub_path = choose_base_prediction()
    meta_oof = load_frame(META_OOF)
    meta_sub = load_frame(META_SUB)
    raw_oof = load_frame(RAW_OOF)
    raw_sub = load_frame(RAW_SUB)

    print(f'[v51] base={base_tag}')
    base_loss, base_per_target = evaluate_frame(train, base_oof)
    meta_loss, meta_per_target = evaluate_frame(train, meta_oof)

    thresholds = {
        target: float(np.quantile(np.abs(meta_oof[target] - base_oof[target]), 0.75))
        for target in TARGETS
    }
    loose_thresholds = {
        target: float(np.quantile(np.abs(meta_oof[target] - base_oof[target]), 0.88))
        for target in TARGETS
    }

    candidates = []
    direct_specs = {
        'v51_meta_q1s1s2_only': {
            'Q1': 0.18, 'Q2': 0.0, 'Q3': 0.0,
            'S1': 0.22, 'S2': 0.20, 'S3': 0.0, 'S4': 0.0,
        },
        'v51_meta_q2s4_probe': {
            'Q1': 0.04, 'Q2': 0.16, 'Q3': 0.02,
            'S1': 0.04, 'S2': 0.04, 'S3': 0.0, 'S4': 0.18,
        },
        'v51_meta_oofadaptive_capped': {
            'Q1': 0.18, 'Q2': 0.18, 'Q3': 0.06,
            'S1': 0.12, 'S2': 0.12, 'S3': 0.0, 'S4': 0.22,
        },
        'v51_meta_guarded_public_safe_direct': {
            'Q1': 0.10, 'Q2': 0.08, 'Q3': 0.03,
            'S1': 0.12, 'S2': 0.12, 'S3': 0.0, 'S4': 0.08,
        },
    }
    for name, weights in direct_specs.items():
        oof_pred = blend_target_weights(base_oof, meta_oof, weights)
        sub_pred = blend_target_weights(base_sub, meta_sub, weights)
        candidates.append(save_candidate(name, train, sub, base_sub, oof_pred, sub_pred, {
            'type': 'direct_capped_base_meta_blend',
            'base_tag': base_tag,
            'weights': weights,
        }))

    guarded_specs = {
        'v51_meta_disagreement_guard_safe': {
            'Q1': 0.14, 'Q2': 0.12, 'Q3': 0.04,
            'S1': 0.16, 'S2': 0.15, 'S3': 0.0, 'S4': 0.12,
        },
        'v51_meta_disagreement_guard_mid': {
            'Q1': 0.22, 'Q2': 0.16, 'Q3': 0.05,
            'S1': 0.22, 'S2': 0.20, 'S3': 0.0, 'S4': 0.18,
        },
        'v51_meta_disagreement_guard_s4probe': {
            'Q1': 0.12, 'Q2': 0.14, 'Q3': 0.04,
            'S1': 0.12, 'S2': 0.12, 'S3': 0.0, 'S4': 0.28,
        },
    }
    for name, high_weights in guarded_specs.items():
        oof_pred, oof_diag = disagreement_guard(base_oof, meta_oof, high_weights, thresholds, low_scale=0.25)
        sub_pred, sub_diag = disagreement_guard(base_sub, meta_sub, high_weights, thresholds, low_scale=0.25)
        candidates.append(save_candidate(name, train, sub, base_sub, oof_pred, sub_pred, {
            'type': 'meta_base_disagreement_guard',
            'base_tag': base_tag,
            'high_weights': high_weights,
            'threshold_source': 'train_q75_abs_meta_minus_base',
            'oof_diag': oof_diag,
            'sub_diag': sub_diag,
        }))

    agreement_specs = {
        'v51_meta_raw_agreement_gate_safe': {
            'Q1': 0.18, 'Q2': 0.14, 'Q3': 0.04,
            'S1': 0.18, 'S2': 0.16, 'S3': 0.0, 'S4': 0.14,
        },
        'v51_meta_raw_agreement_gate_mid': {
            'Q1': 0.28, 'Q2': 0.18, 'Q3': 0.05,
            'S1': 0.26, 'S2': 0.24, 'S3': 0.0, 'S4': 0.20,
        },
        'v51_meta_raw_agreement_gate_loose': {
            'Q1': 0.32, 'Q2': 0.22, 'Q3': 0.06,
            'S1': 0.30, 'S2': 0.28, 'S3': 0.0, 'S4': 0.24,
        },
    }
    for name, high_weights in agreement_specs.items():
        th = loose_thresholds if name.endswith('loose') else thresholds
        oof_pred, oof_diag = raw_meta_agreement_guard(
            base_oof, meta_oof, raw_oof, high_weights, th, low_weight=0.0, disagree_scale=0.20
        )
        sub_pred, sub_diag = raw_meta_agreement_guard(
            base_sub, meta_sub, raw_sub, high_weights, th, low_weight=0.0, disagree_scale=0.20
        )
        candidates.append(save_candidate(name, train, sub, base_sub, oof_pred, sub_pred, {
            'type': 'raw_meta_same_direction_gate',
            'base_tag': base_tag,
            'high_weights': high_weights,
            'threshold_source': 'train_abs_meta_minus_base',
            'oof_diag': oof_diag,
            'sub_diag': sub_diag,
        }))

    candidates = sorted(candidates, key=lambda item: item['oof_loss'])
    summary = {
        'exp_tag': 'v51_meta_guarded_policy',
        'base': {
            'tag': base_tag,
            'oof_path': base_oof_path,
            'submission_path': base_sub_path,
            'oof_loss': base_loss,
            'oof_per_target': base_per_target,
        },
        'meta': {
            'oof_path': str(META_OOF),
            'submission_path': str(META_SUB),
            'oof_loss': meta_loss,
            'oof_per_target': meta_per_target,
            'distribution_vs_base': describe_vs_base(meta_sub, base_sub),
        },
        'thresholds_q75_abs_meta_minus_base': thresholds,
        'thresholds_q88_abs_meta_minus_base': loose_thresholds,
        'candidates': candidates,
        'notes': [
            'v51 is a public-risk probe. Lower OOF is not enough; check distribution_vs_base.',
            'S3 is deliberately zeroed because v50 meta was worse than base on S3.',
            'If any v51 candidate improves public, sequence meta routing is worth a v52 aggressive version.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v51_meta_guarded_policy.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('[v51] base_oof:', f'{base_loss:.6f}')
    print('[v51] meta_oof:', f'{meta_loss:.6f}')
    print('[v51] top candidates by OOF:')
    for item in candidates[:10]:
        print(' ', item['name'], f"oof={item['oof_loss']:.6f}", item['submission'])
    print('[v51] summary:', summary_path)


if __name__ == '__main__':
    main()

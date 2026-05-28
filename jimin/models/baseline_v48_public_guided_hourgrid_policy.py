# v48: public-guided policy layer for the v47 hour-grid residual model.
#
# This script does not retrain. It takes the v47 raw predictions and the
# current v45 anchor, then writes conservative blend candidates. The purpose is
# to use the public feedback from v47 non-S4 w05/w15 to bracket the useful
# amount of raw-sensor signal before spending time on a deeper v49 model.
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
NON_S4_TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_OOF = OOF_DIR / 'oof_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
ANCHOR_SUB = SUB_DIR / 'submission_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
RAW_OOF = OOF_DIR / 'oof_v47_hourgrid_subject_state_residual_raw.csv'
RAW_SUB = SUB_DIR / 'submission_v47_hourgrid_subject_state_residual_raw.csv'

# Known public scores. Lower is better.
PUBLIC_CURVE_POINTS = {
    0.00: 0.5831903668,  # v45 w65 anchor
    0.05: 0.5830874527,  # v47 non-S4 w05
    0.15: 0.5832548135,  # v47 non-S4 w15
}
PUBLIC_TARGET_POLICY_POINTS = {
    'v48_public_curve_non_s4_w066': 0.5830810646,
    'v48_target_delta_scaled_avg07': 0.5825356465,
}


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    return pd.read_csv(path)


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
    return df[['subject_id', 'sleep_date', 'lifelog_date']].copy()


def fit_public_quadratic():
    xs = np.array(sorted(PUBLIC_CURVE_POINTS), dtype=float)
    ys = np.array([PUBLIC_CURVE_POINTS[x] for x in xs], dtype=float)
    a, b, c = np.polyfit(xs, ys, 2)
    if a <= 0:
        opt = 0.07
    else:
        opt = float(np.clip(-b / (2.0 * a), 0.0, 0.15))
    return {
        'points': {f'{k:.2f}': v for k, v in PUBLIC_CURVE_POINTS.items()},
        'quadratic': {'a': float(a), 'b': float(b), 'c': float(c)},
        'estimated_best_weight': opt,
        'estimated_best_score': float(a * opt * opt + b * opt + c),
    }


def blend_with_weights(anchor, raw, weights):
    out = anchor.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        if weight <= 0.0:
            continue
        out[target] = clip_prob((1.0 - weight) * anchor[target] + weight * raw[target])
    return out


def blend_guarded(anchor, raw, high_weights, low_weight, train_thresholds):
    out = anchor.copy()
    thresholds = {}
    active_rates = {}
    for target in TARGETS:
        high_weight = float(high_weights.get(target, 0.0))
        if high_weight <= 0.0:
            continue
        delta = np.abs(raw[target].to_numpy(dtype=float) - anchor[target].to_numpy(dtype=float))
        threshold = float(train_thresholds[target])
        row_weight = np.where(delta <= threshold, high_weight, low_weight)
        out[target] = clip_prob((1.0 - row_weight) * anchor[target] + row_weight * raw[target])
        thresholds[target] = threshold
        active_rates[target] = float(np.mean(row_weight == high_weight))
    return out, thresholds, active_rates


def describe_vs_anchor(pred, anchor):
    pred_arr = pred[TARGETS].to_numpy(dtype=float)
    anchor_arr = anchor[TARGETS].to_numpy(dtype=float)
    diff = pred_arr - anchor_arr
    corr = np.corrcoef(pred_arr.ravel(), anchor_arr.ravel())[0, 1]
    return {
        'corr_vs_anchor': float(corr),
        'mad_vs_anchor': float(np.mean(np.abs(diff))),
        'max_abs_vs_anchor': float(np.max(np.abs(diff))),
        'per_target_mad': {
            target: float(np.mean(np.abs(pred[target] - anchor[target])))
            for target in TARGETS
        },
        'means': {
            target: float(pred[target].mean())
            for target in TARGETS
        },
    }


def save_candidate(name, train, sub, anchor_oof, anchor_sub, oof_pred, sub_pred, policy):
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
        'distribution_vs_anchor': describe_vs_anchor(submission, anchor_sub),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
    }


def target_delta_scaled_weights(train, anchor_oof, raw_oof, avg_weight=0.07, cap=0.10, floor=0.03):
    gains = {}
    for target in NON_S4_TARGETS:
        anchor_loss = target_logloss(train[target], anchor_oof[target])
        raw_loss = target_logloss(train[target], raw_oof[target])
        gains[target] = max(0.0, anchor_loss - raw_loss)
    mean_gain = np.mean(list(gains.values()))
    weights = {}
    for target in TARGETS:
        if target == 'S4':
            weights[target] = 0.0
        else:
            scaled = avg_weight * gains[target] / mean_gain if mean_gain > 0 else avg_weight
            weights[target] = float(np.clip(scaled, floor, cap))
    return weights, gains


def cap_weights(weights, caps):
    capped = dict(weights)
    for target, cap in caps.items():
        capped[target] = min(float(capped.get(target, 0.0)), float(cap))
    return capped


def main():
    ensure_dirs()
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_SAMPLE_PATH)
    anchor_oof = load_frame(ANCHOR_OOF)
    anchor_sub = load_frame(ANCHOR_SUB)
    raw_oof = load_frame(RAW_OOF)
    raw_sub = load_frame(RAW_SUB)

    public_fit = fit_public_quadratic()
    public_weight = public_fit['estimated_best_weight']

    anchor_loss, anchor_per_target = evaluate_frame(train, anchor_oof)
    raw_loss, raw_per_target = evaluate_frame(train, raw_oof)
    delta_scaled_weights, target_oof_gains = target_delta_scaled_weights(
        train,
        anchor_oof,
        raw_oof,
        avg_weight=0.07,
        cap=0.10,
        floor=0.03,
    )
    delta_scaled_variants = {}
    variant_specs = [
        ('v48_target_delta_scaled_avg075_cap11', 0.075, 0.11, 0.025),
        ('v48_target_delta_scaled_avg080_cap12', 0.080, 0.12, 0.025),
        ('v48_target_delta_scaled_avg085_cap12', 0.085, 0.12, 0.025),
        ('v48_target_delta_scaled_avg090_cap13', 0.090, 0.13, 0.025),
        ('v48_target_delta_scaled_avg100_cap14', 0.100, 0.14, 0.025),
    ]
    for name, avg_weight, cap, floor in variant_specs:
        weights, _ = target_delta_scaled_weights(
            train,
            anchor_oof,
            raw_oof,
            avg_weight=avg_weight,
            cap=cap,
            floor=floor,
        )
        delta_scaled_variants[name] = weights

    delta_scaled_variants['v48_target_delta_scaled_avg085_q3guard'] = cap_weights(
        delta_scaled_variants['v48_target_delta_scaled_avg085_cap12'],
        {'Q3': 0.038},
    )
    delta_scaled_variants['v48_target_delta_scaled_avg090_q3s3guard'] = cap_weights(
        delta_scaled_variants['v48_target_delta_scaled_avg090_cap13'],
        {'Q3': 0.040, 'S3': 0.066},
    )
    delta_scaled_variants['v48_target_delta_scaled_avg100_q3s3guard'] = cap_weights(
        delta_scaled_variants['v48_target_delta_scaled_avg100_cap14'],
        {'Q3': 0.040, 'S3': 0.070},
    )

    candidates = []

    uniform_weights = [
        ('v48_public_curve_non_s4_w06', 0.06),
        ('v48_public_curve_non_s4_w066', public_weight),
        ('v48_public_curve_non_s4_w07', 0.07),
        ('v48_public_curve_non_s4_w08', 0.08),
        ('v48_public_curve_non_s4_w09', 0.09),
    ]
    for name, weight in uniform_weights:
        weights = {target: weight for target in NON_S4_TARGETS}
        weights['S4'] = 0.0
        oof_pred = blend_with_weights(anchor_oof, raw_oof, weights)
        sub_pred = blend_with_weights(anchor_sub, raw_sub, weights)
        candidates.append(save_candidate(
            name,
            train,
            sub,
            anchor_oof,
            anchor_sub,
            oof_pred,
            sub_pred,
            {'type': 'uniform_non_s4', 'weights': weights},
        ))

    target_policies = {
        'v48_target_delta_scaled_avg07': delta_scaled_weights,
        **delta_scaled_variants,
        'v48_target_safe_scaled_avg065': {
            'Q1': 0.08, 'Q2': 0.06, 'Q3': 0.04,
            'S1': 0.08, 'S2': 0.08, 'S3': 0.06, 'S4': 0.0,
        },
        'v48_target_sleep_core_avg067': {
            'Q1': 0.07, 'Q2': 0.05, 'Q3': 0.03,
            'S1': 0.09, 'S2': 0.09, 'S3': 0.07, 'S4': 0.0,
        },
        'v48_target_no_q3_avg06': {
            'Q1': 0.08, 'Q2': 0.06, 'Q3': 0.00,
            'S1': 0.08, 'S2': 0.08, 'S3': 0.06, 'S4': 0.0,
        },
    }
    for name, weights in target_policies.items():
        oof_pred = blend_with_weights(anchor_oof, raw_oof, weights)
        sub_pred = blend_with_weights(anchor_sub, raw_sub, weights)
        candidates.append(save_candidate(
            name,
            train,
            sub,
            anchor_oof,
            anchor_sub,
            oof_pred,
            sub_pred,
            {'type': 'target_weighted_non_s4', 'weights': weights},
        ))

    train_thresholds = {
        target: float(np.quantile(np.abs(raw_oof[target] - anchor_oof[target]), 0.85))
        for target in NON_S4_TARGETS
    }
    high_weights = {target: 0.09 for target in NON_S4_TARGETS}
    high_weights['S4'] = 0.0
    guarded_oof, guarded_thresholds, guarded_oof_active = blend_guarded(
        anchor_oof,
        raw_oof,
        high_weights,
        low_weight=0.03,
        train_thresholds=train_thresholds,
    )
    guarded_sub, _, guarded_sub_active = blend_guarded(
        anchor_sub,
        raw_sub,
        high_weights,
        low_weight=0.03,
        train_thresholds=train_thresholds,
    )
    candidates.append(save_candidate(
        'v48_guarded_non_s4_w09_q85_floor03',
        train,
        sub,
        anchor_oof,
        anchor_sub,
        guarded_oof,
        guarded_sub,
        {
            'type': 'rowwise_disagreement_guard',
            'high_weights': high_weights,
            'low_weight': 0.03,
            'train_abs_delta_threshold_q85': guarded_thresholds,
            'oof_high_weight_rate': guarded_oof_active,
            'submission_high_weight_rate': guarded_sub_active,
        },
    ))

    candidates = sorted(candidates, key=lambda item: item['oof_loss'])
    summary = {
        'exp_tag': 'v48_public_guided_hourgrid_policy',
        'inputs': {
            'anchor_oof': str(ANCHOR_OOF),
            'anchor_submission': str(ANCHOR_SUB),
            'raw_oof': str(RAW_OOF),
            'raw_submission': str(RAW_SUB),
        },
        'known_public_scores': PUBLIC_CURVE_POINTS,
        'known_public_target_policy_scores': PUBLIC_TARGET_POLICY_POINTS,
        'public_curve_fit': public_fit,
        'anchor_oof': {'loss': anchor_loss, 'per_target': anchor_per_target},
        'raw_oof': {'loss': raw_loss, 'per_target': raw_per_target},
        'target_oof_gains_anchor_minus_raw': target_oof_gains,
        'candidates': candidates,
        'submission_recommendation': {
            'score_exploit_first': 'submission_v48_target_delta_scaled_avg085_cap12.csv',
            'direction_probe_second': 'submission_v48_target_delta_scaled_avg100_q3s3guard.csv',
            'reason': (
                'The target-delta avg07 candidate beat the uniform public curve by a large margin. '
                'The next useful probe is a moderate strength increase; if that still improves, test the guarded high-strength variant.'
            ),
        },
    }
    summary_path = SUMMARY_DIR / 'summary_v48_public_guided_hourgrid_policy.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('[v48] anchor_oof:', f'{anchor_loss:.6f}')
    print('[v48] raw_oof:', f'{raw_loss:.6f}')
    print('[v48] public fitted best weight:', f'{public_weight:.4f}')
    print('[v48] top candidates by OOF:')
    for item in candidates[:6]:
        print(' ', item['name'], f"oof={item['oof_loss']:.6f}", item['submission'])
    print('[v48] summary:', summary_path)


if __name__ == '__main__':
    main()

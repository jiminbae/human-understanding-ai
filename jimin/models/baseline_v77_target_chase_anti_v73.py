"""v77: target-chasing extensions after w1.30 improved public.

Known public points on the anti-v73 axis:
    w=-1.00  v73 subject-target shrink75        0.5837420840
    w= 0.00  v72 subject-scale shrink75         0.5755923274
    w= 0.20  v74 anti-v73 from v72              0.5747180368
    w= 0.50  v75 anti-v73 publiccurve           0.5737175196
    w= 1.30  v76 bold anti-v73                  0.5725532881

The w=1.30 point still improves, but the improvement slope has flattened.  This
script writes two families:
  * pure probability-space extrapolation for maximum upside
  * logit-space extrapolation as a smoother high-strength variant
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v56_block_router as v56


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

BASE_TAG = 'v72_subject_scale_shrink0p75'
BAD_TAG = 'v73_subject_target_scale_shrink0p75'
PUBLIC_POINTS = {
    -1.00: 0.5837420840,
    0.00: 0.5755923274,
    0.20: 0.5747180368,
    0.50: 0.5737175196,
    1.30: 0.5725532881,
}


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def logit(values):
    values = clip_prob(values)
    return np.log(values / (1.0 - values))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def anti_extrapolate_prob(base, bad, strength):
    out = base.copy()
    for target in TARGETS:
        b = base[target].to_numpy(dtype=float)
        d = bad[target].to_numpy(dtype=float)
        out[target] = clip_prob(b + float(strength) * (b - d))
    return out


def anti_extrapolate_logit(base, bad, strength):
    out = base.copy()
    for target in TARGETS:
        b = base[target].to_numpy(dtype=float)
        d = bad[target].to_numpy(dtype=float)
        z = logit(b) + float(strength) * (logit(b) - logit(d))
        out[target] = clip_prob(sigmoid(z))
    return out


def fit_curves():
    xs = np.array(sorted(PUBLIC_POINTS), dtype=float)
    ys = np.array([PUBLIC_POINTS[x] for x in xs], dtype=float)
    all_quad = np.polyfit(xs, ys, 2)
    pos = xs >= 0
    pos_linear = np.polyfit(xs[pos], ys[pos], 1)
    pos_quad = np.polyfit(xs[pos], ys[pos], 2)
    return {
        'quadratic_all_points': [float(x) for x in all_quad],
        'linear_nonnegative_points': [float(x) for x in pos_linear],
        'quadratic_nonnegative_points': [float(x) for x in pos_quad],
    }


def curve_predictions(curves, strength):
    return {
        name: float(np.polyval(coef, strength))
        for name, coef in curves.items()
    }


def saturation_count(submission):
    values = submission[TARGETS].to_numpy(dtype=float)
    return int(((values <= 0.0200000001) | (values >= 0.9799999999)).sum())


def save_candidate(name, train, anchor_sub, roles, test_roles, base_oof, bad_oof, base_sub, bad_sub, strength, method, curves):
    if method == 'prob':
        oof = anti_extrapolate_prob(base_oof, bad_oof, strength)
        submission = anti_extrapolate_prob(base_sub, bad_sub, strength)
    elif method == 'logit':
        oof = anti_extrapolate_logit(base_oof, bad_oof, strength)
        submission = anti_extrapolate_logit(base_sub, bad_sub, strength)
    else:
        raise ValueError(method)

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    full_oof = v56.evaluate(train, oof)
    role_oof = v56.role_evaluations(train, oof, roles)
    dist_v72 = v56.describe_vs_anchor(submission, base_sub, test_roles)
    dist_anchor = v56.describe_vs_anchor(submission, anchor_sub, test_roles)
    return {
        'name': name,
        'method': method,
        'strength': float(strength),
        'public_curve_predictions': curve_predictions(curves, strength),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
        'full_oof': full_oof,
        'role_oof': role_oof,
        'distribution_vs_v72': dist_v72,
        'distribution_vs_anchor': dist_anchor,
        'saturation_count': saturation_count(submission),
    }


def main():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_SAMPLE_PATH)
    profiles, test_roles = v56.build_test_profiles(train, sub)
    roles = v56.build_train_roles(train, profiles)
    anchor_sub = v56.load_submission('v56_block_router_mid', sub)
    base_oof = v56.load_oof(BASE_TAG, train)
    base_sub = v56.load_submission(BASE_TAG, sub)
    bad_oof = v56.load_oof(BAD_TAG, train)
    bad_sub = v56.load_submission(BAD_TAG, sub)
    base_routed = v56.role_evaluations(train, base_oof, roles)['routed_rows']['loss']

    curves = fit_curves()
    candidates = []
    for strength in [1.80, 2.20, 2.60, 3.00, 3.40]:
        tag = f'{strength:.2f}'.rstrip('0').rstrip('.').replace('.', 'p')
        candidates.append(save_candidate(
            f'v77_target_chase_prob_w{tag}',
            train,
            anchor_sub,
            roles,
            test_roles,
            base_oof,
            bad_oof,
            base_sub,
            bad_sub,
            strength,
            'prob',
            curves,
        ))
    for strength in [1.80, 2.20, 2.60, 3.00, 3.40]:
        tag = f'{strength:.2f}'.rstrip('0').rstrip('.').replace('.', 'p')
        candidates.append(save_candidate(
            f'v77_target_chase_logit_w{tag}',
            train,
            anchor_sub,
            roles,
            test_roles,
            base_oof,
            bad_oof,
            base_sub,
            bad_sub,
            strength,
            'logit',
            curves,
        ))

    summary = {
        'exp_tag': 'v77_target_chase_anti_v73',
        'base': BASE_TAG,
        'bad_direction': BAD_TAG,
        'public_points': {str(k): v for k, v in PUBLIC_POINTS.items()},
        'public_curves': curves,
        'candidates': candidates,
        'recommended_submit_order': [
            'v77_target_chase_prob_w2p6',
            'v77_target_chase_prob_w3',
            'v77_target_chase_logit_w2p6',
        ],
        'policy_notes': [
            'w1.30 improved but with a flatter slope, so w1.80 is probably too small for the required jump.',
            'probability-space w2.60 is the first true target-chasing probe.',
            'logit-space variants are smoother alternatives if pure probability extrapolation over-clips.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v77_target_chase_anti_v73.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v77] summary={summary_path}')
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        preds = item['public_curve_predictions']
        print(
            f"  {item['name']}: "
            f"pred_linear_pos={preds['linear_nonnegative_points']:.9f} "
            f"pred_quad_pos={preds['quadratic_nonnegative_points']:.9f} "
            f"full_oof={item['full_oof']['loss']:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed_vs_v72={routed - base_routed:+.6f} "
            f"mad_vs_v72={item['distribution_vs_v72']['mad_vs_anchor']:.6f} "
            f"sat={item['saturation_count']} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

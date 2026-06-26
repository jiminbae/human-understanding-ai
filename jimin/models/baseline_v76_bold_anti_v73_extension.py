"""v76: bold extension of the public-confirmed anti-v73 axis.

Public observations:
    w=-1.00  v73 subject-target shrink75        0.5837420840
    w= 0.00  v72 subject-scale shrink75         0.5755923274
    w= 0.20  v74 anti-v73 from v72              0.5747180368
    w= 0.50  v75 anti-v73 publiccurve           0.5737175196

The conservative quadratic fit peaks near w=0.90, but the three observed
non-negative points still improve with a roughly linear slope.  These files are
therefore deliberately bold public extrapolations, meant for the remaining
low-submission-budget phase where a small local gain is not enough.
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
}


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def anti_extrapolate(base, bad, strength):
    out = base.copy()
    for target in TARGETS:
        b = base[target].to_numpy(dtype=float)
        d = bad[target].to_numpy(dtype=float)
        out[target] = clip_prob(b + float(strength) * (b - d))
    return out


def fit_curves():
    xs = np.array(sorted(PUBLIC_POINTS), dtype=float)
    ys = np.array([PUBLIC_POINTS[x] for x in xs], dtype=float)
    quad = np.polyfit(xs, ys, 2)
    cubic = np.polyfit(xs, ys, 3)
    pos_mask = xs >= 0
    pos_linear = np.polyfit(xs[pos_mask], ys[pos_mask], 1)
    quad_opt = -quad[1] / (2.0 * quad[0])
    return {
        'quadratic_all_points': {
            'coef': [float(x) for x in quad],
            'estimated_optimum_w': float(quad_opt),
            'estimated_optimum_public': float(np.polyval(quad, quad_opt)),
        },
        'cubic_all_points': {
            'coef': [float(x) for x in cubic],
        },
        'linear_nonnegative_points': {
            'coef': [float(x) for x in pos_linear],
        },
    }


def curve_predictions(curves, strength):
    return {
        'quadratic_all_points': float(np.polyval(curves['quadratic_all_points']['coef'], strength)),
        'cubic_all_points': float(np.polyval(curves['cubic_all_points']['coef'], strength)),
        'linear_nonnegative_points': float(np.polyval(curves['linear_nonnegative_points']['coef'], strength)),
    }


def save_candidate(name, train, anchor_sub, roles, test_roles, base_oof, bad_oof, base_sub, bad_sub, strength, curves):
    oof = anti_extrapolate(base_oof, bad_oof, strength)
    submission = anti_extrapolate(base_sub, bad_sub, strength)
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
        'strength': float(strength),
        'public_curve_predictions': curve_predictions(curves, strength),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
        'full_oof': full_oof,
        'role_oof': role_oof,
        'distribution_vs_v72': dist_v72,
        'distribution_vs_anchor': dist_anchor,
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
    for strength in [0.90, 1.10, 1.30, 1.50, 1.80, 2.00]:
        tag = f'{strength:.2f}'.rstrip('0').rstrip('.').replace('.', 'p')
        candidates.append(save_candidate(
            f'v76_bold_anti_v73_w{tag}',
            train,
            anchor_sub,
            roles,
            test_roles,
            base_oof,
            bad_oof,
            base_sub,
            bad_sub,
            strength,
            curves,
        ))

    summary = {
        'exp_tag': 'v76_bold_anti_v73_extension',
        'base': BASE_TAG,
        'bad_direction': BAD_TAG,
        'public_points': {str(k): v for k, v in PUBLIC_POINTS.items()},
        'public_curves': curves,
        'candidates': candidates,
        'recommended_submit_order': [
            'v76_bold_anti_v73_w1p3',
            'v76_bold_anti_v73_w1p8',
            'v76_bold_anti_v73_w0p9',
        ],
        'policy_notes': [
            'w0.90 is the conservative quadratic optimum and likely only a small gain.',
            'w1.30 is the first bold point: enough extrapolation to matter without jumping all the way to the 0.568 target.',
            'w1.80 is the target-chasing point if w1.30 improves or the remaining budget requires maximum upside.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v76_bold_anti_v73_extension.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    q = curves['quadratic_all_points']
    print(f'[v76] summary={summary_path}')
    print(
        f"[v76] quadratic_optimum_w={q['estimated_optimum_w']:.3f} "
        f"quadratic_public={q['estimated_optimum_public']:.9f}"
    )
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        preds = item['public_curve_predictions']
        print(
            f"  {item['name']}: "
            f"pred_quad={preds['quadratic_all_points']:.9f} "
            f"pred_linear_pos={preds['linear_nonnegative_points']:.9f} "
            f"full_oof={item['full_oof']['loss']:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed_vs_v72={routed - base_routed:+.6f} "
            f"mad_vs_v72={item['distribution_vs_v72']['mad_vs_anchor']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

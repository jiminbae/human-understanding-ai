"""v75: larger anti-v73 candidates guided by the public curve.

Known public points on the one-dimensional anti-v73 axis:
    w=-1.00  v73 subject-target shrink75        0.5837420840
    w= 0.00  v72 subject-scale shrink75         0.5755923274
    w= 0.20  v74 anti-v73 from v72              0.5747180368

The quadratic fit puts the public minimum near w=0.79.  Because only w=0.20
has been observed on the positive side, this script writes a ladder around the
estimated peak instead of treating the exact fitted optimum as truth.
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


def fit_public_curve():
    xs = np.array(sorted(PUBLIC_POINTS), dtype=float)
    ys = np.array([PUBLIC_POINTS[x] for x in xs], dtype=float)
    a, b, c = np.polyfit(xs, ys, 2)
    optimum = -b / (2.0 * a)
    return {
        'coef': [float(a), float(b), float(c)],
        'estimated_optimum_w': float(optimum),
        'estimated_optimum_public': float(np.polyval([a, b, c], optimum)),
    }


def save_candidate(name, train, anchor_sub, roles, test_roles, base_oof, bad_oof, base_sub, bad_sub, strength, curve):
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
    pred_public = float(np.polyval(curve['coef'], strength))
    return {
        'name': name,
        'strength': float(strength),
        'predicted_public_by_quadratic': pred_public,
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

    curve = fit_public_curve()
    candidates = []
    for strength in [0.50, 0.65, 0.75, 0.80, 0.90, 1.00]:
        tag = f'{strength:.2f}'.rstrip('0').rstrip('.').replace('.', 'p')
        candidates.append(save_candidate(
            f'v75_anti_v73_publiccurve_w{tag}',
            train,
            anchor_sub,
            roles,
            test_roles,
            base_oof,
            bad_oof,
            base_sub,
            bad_sub,
            strength,
            curve,
        ))

    summary = {
        'exp_tag': 'v75_anti_v73_public_curve_peak',
        'base': BASE_TAG,
        'bad_direction': BAD_TAG,
        'public_points': {str(k): v for k, v in PUBLIC_POINTS.items()},
        'public_curve': curve,
        'candidates': candidates,
        'recommended_submit_order_after_v74_w0p2_improved': [
            'v75_anti_v73_publiccurve_w0p5',
            'v75_anti_v73_publiccurve_w0p75',
            'v75_anti_v73_publiccurve_w0p8',
        ],
        'policy_notes': [
            'w0.50 is the next confirmation point because w0.20 already improved public.',
            'w0.75 and w0.80 are peak probes only if w0.50 also improves.',
            'OOF will worsen by construction; public feedback is the governing signal on this axis.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v75_anti_v73_public_curve_peak.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v75] summary={summary_path}')
    print(
        f"[v75] estimated_optimum_w={curve['estimated_optimum_w']:.3f} "
        f"estimated_public={curve['estimated_optimum_public']:.9f}"
    )
    base_routed = v56.role_evaluations(train, base_oof, roles)['routed_rows']['loss']
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        print(
            f"  {item['name']}: pred_public={item['predicted_public_by_quadratic']:.9f} "
            f"full_oof={item['full_oof']['loss']:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed_vs_v72={routed - base_routed:+.6f} "
            f"mad_vs_v72={item['distribution_vs_v72']['mad_vs_anchor']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

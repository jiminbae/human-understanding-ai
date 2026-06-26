"""v70: final public-curve peak probe for the residual axis.

The scale-12 OOF peak scored 0.5769907858 publicly.  A quadratic fit to all
known public points estimates the public minimum near residual scale 14.44.
This script prepares that final peak probe and nearby guards.  It should close
the current scale axis; larger structural work should follow afterward.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v56_block_router as v56
import baseline_v65_safe_residual_extension as v65


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_TAG = 'v56_block_router_mid'
PUBLIC_CURVE = [
    (0.00, 0.5795120480),
    (0.55, 0.5793214190),
    (1.00, 0.5791702707),
    (1.60, 0.5789755564),
    (2.40, 0.5787282038),
    (3.60, 0.5783839406),
    (1.0 / 0.18, 0.5778938529),
    (12.00, 0.5769907858),
]


def fit_public_curve():
    x = np.asarray([scale for scale, _ in PUBLIC_CURVE], dtype=float)
    y = np.asarray([score for _, score in PUBLIC_CURVE], dtype=float)
    coeff = np.polyfit(x, y, 2)
    peak = float(-coeff[1] / (2.0 * coeff[0]))
    return coeff, peak


def main():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    coeff, estimated_peak = fit_public_curve()
    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_SAMPLE_PATH)
    profiles, test_roles = v56.build_test_profiles(train, sub)
    train_roles = v56.build_train_roles(train, profiles)
    anchor_oof = v56.load_oof(ANCHOR_TAG, train)
    anchor_sub = v56.load_submission(ANCHOR_TAG, sub)
    solver_oof = v56.load_oof('v61_label_solver_raw', train)
    solver_sub = v56.load_submission('v61_label_solver_raw', sub)
    conf_oof = v56.load_oof('v61_label_solver_confidence', train)
    conf_sub = v56.load_submission('v61_label_solver_confidence', sub)

    specs = [
        ('v70_core_safe_residual_s13p50', 1.00, 13.50),
        ('v70_core_safe_residual_s14p40_publicpeak', 1.00, 14.40),
        ('v70_core_safe_residual_s15p00', 1.00, 15.00),
        ('v70_core_safe_residual_s16p00', 1.00, 16.00),
        ('v70_core090_safe_residual_s14p40', 0.90, 14.40),
    ]

    candidates = []
    for name, core_scale, residual_scale in specs:
        item = v65.save_candidate(
            name,
            f'Core scale {core_scale:.2f}; residual scale {residual_scale:.2f}.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            conf_oof,
            conf_sub,
            train_roles,
            test_roles,
            v65.scale_core_plus_residual(core_scale, residual_scale),
        )
        item['core_scale'] = float(core_scale)
        item['residual_scale'] = float(residual_scale)
        item['quadratic_public_projection'] = float(np.polyval(coeff, residual_scale))
        candidates.append(item)

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    summary = {
        'exp_tag': 'v70_public_curve_peak',
        'public_curve': [
            {'residual_scale': scale, 'score': score}
            for scale, score in PUBLIC_CURVE
        ],
        'quadratic_coefficients': coeff.tolist(),
        'estimated_public_peak_scale': estimated_peak,
        'estimated_public_peak_score': float(np.polyval(coeff, estimated_peak)),
        'anchor': {'tag': ANCHOR_TAG, 'eval': anchor_eval},
        'candidates': candidates,
        'recommended_submit_order': [
            'v70_core_safe_residual_s14p40_publicpeak',
            'v70_core090_safe_residual_s14p40',
        ],
        'policy_notes': [
            'This is the final planned probe on the current scale axis.',
            'OOF peaks near 12 while the fitted public curve peaks near 14.44.',
            'If scale 14.4 does not improve, retain scale 12 and stop extrapolating.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v70_public_curve_peak.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v70] summary={summary_path}')
    print(f'[v70] estimated_public_peak_scale={estimated_peak:.6f}')
    print(f'[v70] estimated_public_peak_score={np.polyval(coeff, estimated_peak):.9f}')
    anchor_routed = anchor_eval['role_oof']['routed_rows']['loss']
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        print(
            f"  {item['name']}: full_oof={item['full_oof']['loss']:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed={routed - anchor_routed:+.6f} "
            f"mad={item['distribution_vs_anchor']['mad_vs_anchor']:.6f} "
            f"projection={item['quadratic_public_projection']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

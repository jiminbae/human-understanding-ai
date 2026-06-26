"""v66: extrapolate the twice-public-validated safe-residual curve.

Known public points on residual scale:
    scale 0.00 (core_mid)  0.5795120480
    scale 0.55             0.5793214190
    scale 1.00             0.5791702707

The two segment slopes are nearly identical, so this experiment prepares the
next larger residual scales while leaving core axes fixed and Q3/S3 frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import baseline_v56_block_router as v56
import baseline_v61_subject_block_label_solver as v61
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
    {'residual_scale': 0.00, 'tag': 'v61_anti_label_core_mid', 'score': 0.579512048},
    {'residual_scale': 0.55, 'tag': 'v64_core_plus_safe_residual_s0p55', 'score': 0.579321419},
    {'residual_scale': 1.00, 'tag': 'v64_core_plus_safe_residual_s1p0', 'score': 0.5791702707},
]


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def segment_slopes():
    slopes = []
    for left, right in zip(PUBLIC_CURVE[:-1], PUBLIC_CURVE[1:]):
        slopes.append(
            (right['score'] - left['score'])
            / (right['residual_scale'] - left['residual_scale'])
        )
    return slopes


def projected_score(scale: float):
    last = PUBLIC_CURVE[-1]
    slope = segment_slopes()[-1]
    return float(last['score'] + (float(scale) - last['residual_scale']) * slope)


def main():
    ensure_dirs()
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
        ('v66_core_safe_residual_s1p85', 1.00, 1.85),
        ('v66_core_safe_residual_s2p10', 1.00, 2.10),
        ('v66_core_safe_residual_s2p40', 1.00, 2.40),
        ('v66_core_safe_residual_s2p80', 1.00, 2.80),
        ('v66_core090_safe_residual_s2p10', 0.90, 2.10),
        ('v66_core080_safe_residual_s2p40', 0.80, 2.40),
    ]

    candidates = []
    for name, core_scale, residual_scale in specs:
        item = v65.save_candidate(
            name,
            f'Core scale {core_scale:.2f}; public-validated safe residual scale {residual_scale:.2f}.',
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
        item['linear_public_projection'] = projected_score(residual_scale)
        candidates.append(item)

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    summary = {
        'exp_tag': 'v66_residual_curve_extrapolation',
        'public_curve': PUBLIC_CURVE,
        'segment_slopes': segment_slopes(),
        'anchor': {'tag': ANCHOR_TAG, 'eval': anchor_eval},
        'candidates': candidates,
        'recommended_submit_order': [
            'v65_core_safe_residual_s1p60',
            'v66_core_safe_residual_s2p10',
            'v66_core090_safe_residual_s2p10',
        ],
        'policy_notes': [
            'The projection is a diagnostic extrapolation from only three public points.',
            'Submit v65 s1p60 before jumping beyond residual scale 2.0.',
            'Core-soft variants test whether residual gain continues after core axes saturate.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v66_residual_curve_extrapolation.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v66] summary={summary_path}')
    print(f'[v66] public_slopes={segment_slopes()}')
    anchor_routed = anchor_eval['role_oof']['routed_rows']['loss']
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        print(
            f"  {item['name']}: full_oof={item['full_oof']['loss']:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed={routed - anchor_routed:+.6f} "
            f"mad={item['distribution_vs_anchor']['mad_vs_anchor']:.6f} "
            f"projection={item['linear_public_projection']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

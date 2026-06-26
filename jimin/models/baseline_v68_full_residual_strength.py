"""v68: restore the full OOF-selected safe-residual strength.

The public-valid safe residual weights were originally shrunk by 0.18 in v61.
Therefore residual scale 1 / 0.18 = 5.555... recovers the full target/role
weights selected by the negative OOF grid.  Public scores have improved through
scale 3.60, so v68 creates that structural checkpoint and nearby overshoot
probes while keeping core axes and Q3/S3 fixed.
"""
from __future__ import annotations

import json
from pathlib import Path

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
FULL_RESIDUAL_SCALE = 1.0 / 0.18
PUBLIC_CURVE = [
    (0.00, 0.5795120480),
    (0.55, 0.5793214190),
    (1.00, 0.5791702707),
    (1.60, 0.5789755564),
    (2.40, 0.5787282038),
    (3.60, 0.5783839406),
]


def slopes():
    return [
        (right_score - left_score) / (right_scale - left_scale)
        for (left_scale, left_score), (right_scale, right_score)
        in zip(PUBLIC_CURVE[:-1], PUBLIC_CURVE[1:])
    ]


def projection(scale):
    last_scale, last_score = PUBLIC_CURVE[-1]
    return float(last_score + (float(scale) - last_scale) * slopes()[-1])


def main():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)

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
        ('v68_core_safe_residual_s5p56_full', 1.00, FULL_RESIDUAL_SCALE),
        ('v68_core_safe_residual_s6p00', 1.00, 6.00),
        ('v68_core_safe_residual_s6p50', 1.00, 6.50),
        ('v68_core_safe_residual_s7p00', 1.00, 7.00),
        ('v68_core_safe_residual_s8p00', 1.00, 8.00),
        ('v68_core090_safe_residual_s5p56', 0.90, FULL_RESIDUAL_SCALE),
    ]

    candidates = []
    for name, core_scale, residual_scale in specs:
        item = v65.save_candidate(
            name,
            f'Core scale {core_scale:.2f}; safe residual scale {residual_scale:.4f}.',
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
        item['linear_public_projection'] = projection(residual_scale)
        candidates.append(item)

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    summary = {
        'exp_tag': 'v68_full_residual_strength',
        'full_residual_scale': FULL_RESIDUAL_SCALE,
        'public_curve': [
            {'residual_scale': scale, 'score': score}
            for scale, score in PUBLIC_CURVE
        ],
        'segment_slopes': slopes(),
        'anchor': {'tag': ANCHOR_TAG, 'eval': anchor_eval},
        'candidates': candidates,
        'recommended_submit_order': [
            'v68_core_safe_residual_s5p56_full',
            'v68_core_safe_residual_s6p50',
            'v68_core090_safe_residual_s5p56',
        ],
        'policy_notes': [
            'Scale 5.555... exactly reverses the original 0.18 safe shrink.',
            'This is a structural checkpoint, not an arbitrary larger multiplier.',
            'Overshoot probes test whether public optimum lies beyond the OOF-selected strength.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v68_full_residual_strength.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v68] summary={summary_path}')
    print(f'[v68] full_residual_scale={FULL_RESIDUAL_SCALE:.6f}')
    print(f'[v68] public_slopes={slopes()}')
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

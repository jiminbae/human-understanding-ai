"""v65: extend the public-validated safe-residual recombination axis.

v64_core_plus_safe_residual_s0p55 scored 0.579321419, a much larger public gain
than the earlier core_mid step.  That validates the non-overlapping safe
residual axes.  This script prepares stronger variants along the same axis,
while keeping the core axes fixed and Q3/S3 frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import baseline_v56_block_router as v56
import baseline_v61_subject_block_label_solver as v61
import baseline_v64_public_axis_recombination as v64


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_TAG = 'v56_block_router_mid'
PUBLIC_SCORES = {
    'v56_block_router_mid': 0.5798876532,
    'v61_anti_label_oof_selected_safe': 0.579548671,
    'v61_anti_label_core_mid': 0.579512048,
    'v62_anti_core_mid_bold_blend50': 0.5795531685,
    'v64_core_plus_safe_residual_s0p55': 0.579321419,
}


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def scale_core_plus_residual(core_scale: float, residual_scale: float):
    core = v64.scale_weights(v64.CORE_WEIGHTS, core_scale)
    residual = v64.safe_residual_weights(residual_scale)
    return v64.add_weights(core, residual)


def save_candidate(
    name,
    note,
    train,
    anchor_oof,
    anchor_sub,
    solver_oof,
    solver_sub,
    conf_oof,
    conf_sub,
    train_roles,
    test_roles,
    weights,
    cap=0.16,
):
    oof = v61.apply_label_bridge(anchor_oof, solver_oof, conf_oof, train_roles, weights, cap=cap)
    submission = v61.apply_label_bridge(anchor_sub, solver_sub, conf_sub, test_roles, weights, cap=cap)
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    return {
        'name': name,
        'note': note,
        'weights': v61.normalize_weights(weights),
        'cap': float(cap),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
        'full_oof': v56.evaluate(train, oof),
        'role_oof': v56.role_evaluations(train, oof, train_roles),
        'distribution_vs_anchor': v56.describe_vs_anchor(submission, anchor_sub, test_roles),
    }


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
        ('v65_core_safe_residual_s1p15', 1.00, 1.15, 'Core fixed; safe residual at 115%.'),
        ('v65_core_safe_residual_s1p35', 1.00, 1.35, 'Core fixed; safe residual at 135%.'),
        ('v65_core_safe_residual_s1p60', 1.00, 1.60, 'Core fixed; safe residual at 160%.'),
        ('v65_core095_safe_residual_s1p35', 0.95, 1.35, 'Slightly soften core while extending safe residual.'),
        ('v65_core090_safe_residual_s1p60', 0.90, 1.60, 'More conservative core with stronger safe residual.'),
    ]

    candidates = []
    for name, core_scale, residual_scale, note in specs:
        candidates.append(save_candidate(
            name,
            note,
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            conf_oof,
            conf_sub,
            train_roles,
            test_roles,
            scale_core_plus_residual(core_scale, residual_scale),
        ))

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    v64_s055 = v56.load_oof('v64_core_plus_safe_residual_s0p55', train)
    v64_s055_sub = v56.load_submission('v64_core_plus_safe_residual_s0p55', sub)
    baseline = {
        'v64_s0p55': {
            'known_public': PUBLIC_SCORES['v64_core_plus_safe_residual_s0p55'],
            'full_oof': v56.evaluate(train, v64_s055),
            'role_oof': v56.role_evaluations(train, v64_s055, train_roles),
            'distribution_vs_anchor': v56.describe_vs_anchor(v64_s055_sub, anchor_sub, test_roles),
        }
    }
    sorted_candidates = sorted(candidates, key=lambda item: item['full_oof']['loss'])
    summary = {
        'exp_tag': 'v65_safe_residual_extension',
        'public_scores': PUBLIC_SCORES,
        'anchor': {'tag': ANCHOR_TAG, 'eval': anchor_eval},
        'baseline': baseline,
        'candidates': candidates,
        'oof_sorted': [item['name'] for item in sorted_candidates],
        'recommended_submit_order': [
            'v64_core_plus_safe_residual_s1p0',
            'v65_core_safe_residual_s1p35',
            'v65_core095_safe_residual_s1p35',
        ],
        'policy_notes': [
            'Submit v64 s1p0 before v65 if submission budget allows only one more immediate check.',
            'v65 candidates keep the public-positive recombination structure and extend only the safe residual strength.',
            'Core-soft variants are included in case core_mid is near saturation but safe residual still transfers.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v65_safe_residual_extension.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v65] summary={summary_path}')
    anchor_routed = anchor_eval['role_oof']['routed_rows']['loss']
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        full = item['full_oof']['loss']
        mad = item['distribution_vs_anchor']['mad_vs_anchor']
        print(
            f"  {item['name']}: full_oof={full:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed={routed - anchor_routed:+.6f} "
            f"mad={mad:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

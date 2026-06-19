"""v63: local refinement around the public-best v61 core_mid.

Public feedback so far:
    v56_block_router_mid                 0.5798876532
    v61_anti_label_oof_selected_safe     0.579548671
    v61_anti_label_core_mid              0.579512048
    v62_anti_core_mid_bold_blend50       0.5795531685

The anti-label sign is validated, but the bold direction oversteps.  This file
creates tiny local scale variants around core_mid plus axis ablations to learn
which part of core_mid is carrying the public gain.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import baseline_v56_block_router as v56
import baseline_v61_subject_block_label_solver as v61


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
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
}

CORE_MID_WEIGHTS = {
    'simple_interior': {'Q2': -0.24},
    'fragmented_interior': {'S1': -0.34, 'S2': -0.34},
    'tail': {'Q1': -0.25, 'S4': -0.25},
}

CORE_AXIS_WEIGHTS = {
    'v63_core_simple_q2_only': {
        'simple_interior': {'Q2': -0.24},
    },
    'v63_core_fragmented_s1s2_only': {
        'fragmented_interior': {'S1': -0.34, 'S2': -0.34},
    },
    'v63_core_tail_q1s4_only': {
        'tail': {'Q1': -0.25, 'S4': -0.25},
    },
    'v63_core_drop_simple_q2': {
        'fragmented_interior': {'S1': -0.34, 'S2': -0.34},
        'tail': {'Q1': -0.25, 'S4': -0.25},
    },
    'v63_core_drop_fragmented_s1s2': {
        'simple_interior': {'Q2': -0.24},
        'tail': {'Q1': -0.25, 'S4': -0.25},
    },
    'v63_core_drop_tail_q1s4': {
        'simple_interior': {'Q2': -0.24},
        'fragmented_interior': {'S1': -0.34, 'S2': -0.34},
    },
}


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def scale_weights(weights: dict, scale: float):
    out = {}
    for role, spec in weights.items():
        out[role] = {target: float(value) * float(scale) for target, value in spec.items()}
    return out


def save_candidate(name, train, anchor_oof, anchor_sub, solver_oof, solver_sub, conf_oof, conf_sub, train_roles, test_roles, weights, cap):
    oof = v61.apply_label_bridge(anchor_oof, solver_oof, conf_oof, train_roles, weights, cap=cap)
    submission = v61.apply_label_bridge(anchor_sub, solver_sub, conf_sub, test_roles, weights, cap=cap)
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    return {
        'name': name,
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

    specs = []
    for scale in [0.92, 0.97, 1.03, 1.06, 1.10]:
        tag = str(scale).replace('.', 'p')
        specs.append((f'v63_core_mid_scale{tag}', scale_weights(CORE_MID_WEIGHTS, scale), 0.160))
    for name, weights in CORE_AXIS_WEIGHTS.items():
        specs.append((name, weights, 0.160))

    candidates = [
        save_candidate(
            name,
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
            cap,
        )
        for name, weights, cap in specs
    ]

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    core_oof = v56.load_oof('v61_anti_label_core_mid', train)
    core_eval = {
        'full_oof': v56.evaluate(train, core_oof),
        'role_oof': v56.role_evaluations(train, core_oof, train_roles),
    }
    sorted_candidates = sorted(candidates, key=lambda item: item['full_oof']['loss'])
    summary = {
        'exp_tag': 'v63_core_mid_local_refine',
        'public_scores': PUBLIC_SCORES,
        'anchor': {'tag': ANCHOR_TAG, 'eval': anchor_eval},
        'core_mid': {
            'tag': 'v61_anti_label_core_mid',
            'known_public': PUBLIC_SCORES['v61_anti_label_core_mid'],
            'eval': core_eval,
        },
        'candidates': candidates,
        'oof_sorted': [item['name'] for item in sorted_candidates],
        'recommended_submit_order': [
            'v63_core_mid_scale1p03',
            'v63_core_drop_tail_q1s4',
            'v63_core_mid_scale0p97',
        ],
        'policy_notes': [
            'The failed blend50 submission says not to move toward bold as a block.',
            'Scale variants are tiny local probes around the current public best.',
            'Axis ablations are diagnostic: they can reveal whether simple Q2, fragmented S1/S2, or tail Q1/S4 is public-toxic.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v63_core_mid_local_refine.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v63] summary={summary_path}')
    anchor_routed = anchor_eval['role_oof']['routed_rows']['loss']
    core_routed = core_eval['role_oof']['routed_rows']['loss']
    print(f'[v63] anchor_full={anchor_eval["full_oof"]["loss"]:.6f} '
          f'core_full={core_eval["full_oof"]["loss"]:.6f} '
          f'core_routed_delta={core_routed - anchor_routed:+.6f}')
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

"""v64: public-feedback axis recombination.

This is the next big step after v61/v62 public feedback:

    v56_block_router_mid                 0.5798876532
    v61_anti_label_oof_selected_safe     0.579548671
    v61_anti_label_core_mid              0.579512048
    v62_anti_core_mid_bold_blend50       0.5795531685

Interpretation:
  * The anti-label sign is public-valid.
  * Broad safe axes are public-valid.
  * Focused core axes are slightly better.
  * Moving toward bold as a block is public-toxic.

So v64 does not scale core upward.  It recombines the two public-positive
families: keep the focused core axes and add only the non-overlapping residual
axes from the safe public-positive model, plus a few convex blends between the
two known-good submissions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v56_block_router as v56
import baseline_v61_subject_block_label_solver as v61


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']
ROLES = ['simple_interior', 'fragmented_interior', 'tail']

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

SAFE_WEIGHTS = {
    'simple_interior': {'Q1': -0.036, 'Q2': -0.153, 'S1': -0.036, 'S2': -0.054},
    'fragmented_interior': {'Q1': -0.054, 'S1': -0.18, 'S2': -0.18, 'S4': -0.054},
    'tail': {'Q1': -0.126, 'Q2': -0.054, 'S1': -0.054, 'S2': -0.072, 'S4': -0.126},
}

CORE_WEIGHTS = {
    'simple_interior': {'Q2': -0.24},
    'fragmented_interior': {'S1': -0.34, 'S2': -0.34},
    'tail': {'Q1': -0.25, 'S4': -0.25},
}


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def empty_weights():
    return {role: {target: 0.0 for target in TARGETS} for role in ROLES}


def normalized(weights: dict):
    out = empty_weights()
    for role, spec in weights.items():
        for target, value in spec.items():
            out[role][target] = float(value)
    return out


def scale_weights(weights: dict, scale: float):
    out = {}
    for role, spec in weights.items():
        out[role] = {target: float(value) * float(scale) for target, value in spec.items()}
    return out


def add_weights(left: dict, right: dict):
    out = normalized(left)
    for role, spec in normalized(right).items():
        for target, value in spec.items():
            out[role][target] += float(value)
    return out


def safe_residual_weights(scale: float):
    out = empty_weights()
    core = normalized(CORE_WEIGHTS)
    safe = normalized(SAFE_WEIGHTS)
    for role in ROLES:
        for target in TARGETS:
            if abs(core[role][target]) > 0:
                continue
            out[role][target] = float(scale) * safe[role][target]
    return out


def core_plus_safe_residual(scale: float):
    return add_weights(CORE_WEIGHTS, safe_residual_weights(scale))


def tailored_public_weights(kind: str):
    if kind == 'frag_sleep_plus_safe':
        # Keep the strongest OOF/public-compatible core block, add safe residual,
        # and soften tail where the bold failure warned us not to push.
        return add_weights(
            {
                'simple_interior': {'Q2': -0.20},
                'fragmented_interior': {'S1': -0.36, 'S2': -0.36},
                'tail': {'Q1': -0.16, 'S4': -0.16},
            },
            safe_residual_weights(0.75),
        )
    if kind == 'tail_soft_core_safe':
        return add_weights(
            {
                'simple_interior': {'Q2': -0.24},
                'fragmented_interior': {'S1': -0.34, 'S2': -0.34},
                'tail': {'Q1': -0.14, 'S4': -0.14},
            },
            safe_residual_weights(1.00),
        )
    if kind == 'fragmented_sleep_safe_only':
        return add_weights(
            {
                'fragmented_interior': {'S1': -0.38, 'S2': -0.38},
            },
            safe_residual_weights(1.00),
        )
    raise ValueError(kind)


def blend_frames(left: pd.DataFrame, right: pd.DataFrame, weight_right: float):
    out = left.copy()
    w = float(weight_right)
    for target in TARGETS:
        out[target] = clip_prob(
            (1.0 - w) * left[target].to_numpy(dtype=float)
            + w * right[target].to_numpy(dtype=float)
        )
    return out


def save_frame(path: Path, frame: pd.DataFrame):
    out = frame.copy()
    out[TARGETS] = out[TARGETS].astype(float).apply(clip_prob)
    out.to_csv(path, index=False)


def save_weight_candidate(
    name: str,
    note: str,
    train: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    solver_oof: pd.DataFrame,
    solver_sub: pd.DataFrame,
    conf_oof: pd.DataFrame,
    conf_sub: pd.DataFrame,
    train_roles: pd.Series,
    test_roles: pd.Series,
    weights: dict,
    cap: float = 0.16,
):
    oof = v61.apply_label_bridge(anchor_oof, solver_oof, conf_oof, train_roles, weights, cap=cap)
    submission = v61.apply_label_bridge(anchor_sub, solver_sub, conf_sub, test_roles, weights, cap=cap)
    return save_candidate(name, note, train, anchor_sub, train_roles, test_roles, oof, submission, {
        'kind': 'weight_recombine',
        'weights': v61.normalize_weights(weights),
        'cap': float(cap),
    })


def save_blend_candidate(
    name: str,
    note: str,
    train: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    train_roles: pd.Series,
    test_roles: pd.Series,
    left_oof: pd.DataFrame,
    left_sub: pd.DataFrame,
    right_oof: pd.DataFrame,
    right_sub: pd.DataFrame,
    weight_right: float,
    left_tag: str,
    right_tag: str,
):
    oof = blend_frames(left_oof, right_oof, weight_right)
    submission = blend_frames(left_sub, right_sub, weight_right)
    return save_candidate(name, note, train, anchor_sub, train_roles, test_roles, oof, submission, {
        'kind': 'convex_public_blend',
        'left_tag': left_tag,
        'right_tag': right_tag,
        'weight_right': float(weight_right),
    })


def save_candidate(
    name: str,
    note: str,
    train: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    train_roles: pd.Series,
    test_roles: pd.Series,
    oof: pd.DataFrame,
    submission: pd.DataFrame,
    spec: dict,
):
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    save_frame(oof_path, oof)
    save_frame(sub_path, submission)
    return {
        'name': name,
        'note': note,
        'spec': spec,
        'oof_path': str(oof_path),
        'submission': str(sub_path),
        'full_oof': v56.evaluate(train, oof),
        'role_oof': v56.role_evaluations(train, oof, train_roles),
        'distribution_vs_anchor': v56.describe_vs_anchor(submission, anchor_sub, test_roles),
    }


def public_diagnostics():
    base = PUBLIC_SCORES['v56_block_router_mid']
    safe = PUBLIC_SCORES['v61_anti_label_oof_selected_safe']
    core = PUBLIC_SCORES['v61_anti_label_core_mid']
    blend = PUBLIC_SCORES['v62_anti_core_mid_bold_blend50']
    return {
        'safe_delta_vs_anchor': safe - base,
        'core_delta_vs_anchor': core - base,
        'core_delta_vs_safe': core - safe,
        'bold_blend_delta_vs_core': blend - core,
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

    safe_oof = v56.load_oof('v61_anti_label_oof_selected_safe', train)
    safe_sub = v56.load_submission('v61_anti_label_oof_selected_safe', sub)
    core_oof = v56.load_oof('v61_anti_label_core_mid', train)
    core_sub = v56.load_submission('v61_anti_label_core_mid', sub)

    candidates = []
    for scale in [0.35, 0.55, 0.75, 1.00]:
        tag = str(scale).replace('.', 'p')
        candidates.append(save_weight_candidate(
            f'v64_core_plus_safe_residual_s{tag}',
            f'Core-mid axes plus {scale:.0%} of non-overlapping safe public-positive residual axes.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            conf_oof,
            conf_sub,
            train_roles,
            test_roles,
            core_plus_safe_residual(scale),
        ))

    for kind in ['frag_sleep_plus_safe', 'tail_soft_core_safe', 'fragmented_sleep_safe_only']:
        candidates.append(save_weight_candidate(
            f'v64_{kind}',
            f'Public-feedback tailored recombination: {kind}.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            conf_oof,
            conf_sub,
            train_roles,
            test_roles,
            tailored_public_weights(kind),
        ))

    for weight_core in [0.35, 0.50, 0.65, 0.80]:
        tag = str(weight_core).replace('.', 'p')
        candidates.append(save_blend_candidate(
            f'v64_public_blend_safe_core_wcore{tag}',
            f'Convex blend of two public-positive submissions: safe and core_mid, core weight {weight_core:.0%}.',
            train,
            anchor_sub,
            train_roles,
            test_roles,
            safe_oof,
            safe_sub,
            core_oof,
            core_sub,
            weight_core,
            'v61_anti_label_oof_selected_safe',
            'v61_anti_label_core_mid',
        ))

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    known_eval = {
        'safe': {
            'full_oof': v56.evaluate(train, safe_oof),
            'role_oof': v56.role_evaluations(train, safe_oof, train_roles),
            'distribution_vs_anchor': v56.describe_vs_anchor(safe_sub, anchor_sub, test_roles),
            'known_public': PUBLIC_SCORES['v61_anti_label_oof_selected_safe'],
        },
        'core_mid': {
            'full_oof': v56.evaluate(train, core_oof),
            'role_oof': v56.role_evaluations(train, core_oof, train_roles),
            'distribution_vs_anchor': v56.describe_vs_anchor(core_sub, anchor_sub, test_roles),
            'known_public': PUBLIC_SCORES['v61_anti_label_core_mid'],
        },
    }
    sorted_candidates = sorted(candidates, key=lambda item: item['full_oof']['loss'])
    summary = {
        'exp_tag': 'v64_public_axis_recombination',
        'public_scores': PUBLIC_SCORES,
        'public_diagnostics': public_diagnostics(),
        'anchor': {'tag': ANCHOR_TAG, 'eval': anchor_eval},
        'known_public_positive': known_eval,
        'candidates': candidates,
        'oof_sorted': [item['name'] for item in sorted_candidates],
        'recommended_submit_order': [
            'v64_core_plus_safe_residual_s0p55',
            'v64_tail_soft_core_safe',
            'v64_public_blend_safe_core_wcore0p65',
        ],
        'policy_notes': [
            'This is a structural recombination, not a local scale-up.',
            'The safest big step is core_mid plus only non-overlapping safe residual axes.',
            'Tail-soft variants are included because the bold blend failure may be warning specifically about over-pushed tail axes.',
            'Convex safe/core blends test whether the two known public-positive submissions have complementary errors.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v64_public_axis_recombination.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v64] summary={summary_path}')
    print(f'[v64] public_diag={public_diagnostics()}')
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

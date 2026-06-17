from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v56_block_router as v56


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_TAG = 'v56_block_router_mid'
SOURCE_TAGS = {
    'v47_raw': 'v47_hourgrid_subject_state_residual_raw',
    'v53_raw': 'v53_split_aware_posterior_raw',
    'v50_raw': 'v50_sequence_meta_raw',
    'v30': 'v30_bidir_history_extended',
}

ROUTE_TEMPLATE = [
    {'role': 'simple_interior', 'target': 'Q1', 'source': 'v53_raw'},
    {'role': 'simple_interior', 'target': 'Q2', 'source': 'v50_raw'},
    {'role': 'simple_interior', 'target': 'Q3', 'source': 'v30'},
    {'role': 'simple_interior', 'target': 'S1', 'source': 'v47_raw'},
    {'role': 'simple_interior', 'target': 'S2', 'source': 'v47_raw'},
    {'role': 'simple_interior', 'target': 'S3', 'source': 'v47_raw'},
    {'role': 'simple_interior', 'target': 'S4', 'source': 'v47_raw'},
    {'role': 'fragmented_interior', 'target': 'Q1', 'source': 'v53_raw'},
    {'role': 'fragmented_interior', 'target': 'Q2', 'source': 'v53_raw'},
    {'role': 'fragmented_interior', 'target': 'Q3', 'source': 'v50_raw'},
    {'role': 'fragmented_interior', 'target': 'S1', 'source': 'v53_raw'},
    {'role': 'fragmented_interior', 'target': 'S2', 'source': 'v53_raw'},
    {'role': 'fragmented_interior', 'target': 'S3', 'source': 'v50_raw'},
    {'role': 'fragmented_interior', 'target': 'S4', 'source': 'v47_raw'},
    {'role': 'tail', 'target': 'Q1', 'source': 'v53_raw'},
    {'role': 'tail', 'target': 'Q2', 'source': 'v53_raw'},
    {'role': 'tail', 'target': 'Q3', 'source': 'v47_raw'},
    {'role': 'tail', 'target': 'S1', 'source': 'v50_raw'},
    {'role': 'tail', 'target': 'S3', 'source': 'v47_raw'},
    {'role': 'tail', 'target': 'S4', 'source': 'v53_raw'},
]


def routes(include_q3s3: bool = True):
    if include_q3s3:
        return list(ROUTE_TEMPLATE)
    return [route for route in ROUTE_TEMPLATE if route['target'] not in {'Q3', 'S3'}]


def rank_remap_values(anchor_values: pd.Series, source_values: pd.Series) -> np.ndarray:
    n = len(anchor_values)
    if n <= 1:
        return anchor_values.to_numpy(dtype=float)
    order = np.argsort(source_values.to_numpy(dtype=float), kind='mergesort')
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.sort(anchor_values.to_numpy(dtype=float))
    return ranked


def apply_centered_delta(anchor, sources, frame, roles, route_spec, strength):
    out = anchor.copy()
    for route in route_spec:
        role = route['role']
        target = route['target']
        mask = roles == role
        if not bool(mask.any()):
            continue
        delta = sources[route['source']].loc[mask, target] - anchor.loc[mask, target]
        centered = delta - float(delta.mean())
        out.loc[mask, target] = v56.clip_prob(anchor.loc[mask, target] + strength * centered)
    return out


def apply_rank_remap(anchor, sources, frame, roles, route_spec, strength):
    out = anchor.copy()
    for route in route_spec:
        role = route['role']
        target = route['target']
        mask = roles == role
        if not bool(mask.any()):
            continue
        remapped = rank_remap_values(anchor.loc[mask, target], sources[route['source']].loc[mask, target])
        delta = remapped - anchor.loc[mask, target].to_numpy(dtype=float)
        out.loc[mask, target] = v56.clip_prob(anchor.loc[mask, target] + strength * delta)
    return out


def apply_block_centered_delta(anchor, sources, frame, roles, route_spec, strength):
    out = anchor.copy()
    for route in route_spec:
        role = route['role']
        target = route['target']
        mask = roles == role
        if not bool(mask.any()):
            continue
        delta = sources[route['source']].loc[mask, target] - anchor.loc[mask, target]
        tmp = pd.DataFrame({
            'subject_id': frame.loc[mask, 'subject_id'].to_numpy(),
            'delta': delta.to_numpy(dtype=float),
        }, index=delta.index)
        block_delta = tmp.groupby('subject_id')['delta'].transform('mean')
        centered = block_delta - float(block_delta.mean())
        out.loc[mask, target] = v56.clip_prob(anchor.loc[mask, target] + strength * centered)
    return out


def apply_candidate(anchor, sources, frame, roles, spec):
    route_spec = routes(include_q3s3=spec.get('include_q3s3', True))
    strength = float(spec['strength'])
    kind = spec['kind']
    if kind == 'centered_delta':
        return apply_centered_delta(anchor, sources, frame, roles, route_spec, strength)
    if kind == 'rank_remap':
        return apply_rank_remap(anchor, sources, frame, roles, route_spec, strength)
    if kind == 'block_centered':
        return apply_block_centered_delta(anchor, sources, frame, roles, route_spec, strength)
    if kind == 'hybrid_rank_block':
        rank = apply_rank_remap(anchor, sources, frame, roles, route_spec, strength)
        block = apply_block_centered_delta(anchor, sources, frame, roles, route_spec, strength)
        out = anchor.copy()
        out[v56.TARGETS] = v56.clip_prob((rank[v56.TARGETS] + block[v56.TARGETS]) / 2.0)
        return out
    raise ValueError(f'Unknown candidate kind: {kind}')


CANDIDATES = {
    'v58_centered_delta_w50': {
        'kind': 'centered_delta',
        'strength': 0.50,
        'include_q3s3': True,
        'note': 'Mean-preserving raw delta at 50% strength.',
    },
    'v58_centered_delta_w100': {
        'kind': 'centered_delta',
        'strength': 1.00,
        'include_q3s3': True,
        'note': 'Mean-preserving raw delta at full strength.',
    },
    'v58_centered_delta_no_q3s3_w100': {
        'kind': 'centered_delta',
        'strength': 1.00,
        'include_q3s3': False,
        'note': 'Mean-preserving raw delta at full strength, with Q3/S3 frozen.',
    },
    'v58_rank_remap_w50': {
        'kind': 'rank_remap',
        'strength': 0.50,
        'include_q3s3': True,
        'note': 'Anchor distribution preserved; source ranks used at 50% strength.',
    },
    'v58_rank_remap_w100': {
        'kind': 'rank_remap',
        'strength': 1.00,
        'include_q3s3': True,
        'note': 'Anchor distribution preserved; source ranks used at full strength.',
    },
    'v58_rank_remap_no_q3s3_w100': {
        'kind': 'rank_remap',
        'strength': 1.00,
        'include_q3s3': False,
        'note': 'Anchor distribution preserved with Q3/S3 frozen.',
    },
    'v58_block_centered_w100': {
        'kind': 'block_centered',
        'strength': 1.00,
        'include_q3s3': True,
        'note': 'Subject-role block-state centered delta at full strength.',
    },
    'v58_hybrid_rank_block_w100': {
        'kind': 'hybrid_rank_block',
        'strength': 1.00,
        'include_q3s3': True,
        'note': 'Average of rank-remap and subject-role block-centered signals.',
    },
}


def save_candidate(name, train, sub, anchor_oof, anchor_sub, source_oofs, source_subs, train_roles, test_roles, spec):
    oof = apply_candidate(anchor_oof, source_oofs, train, train_roles, spec)
    submission = apply_candidate(anchor_sub, source_subs, sub, test_roles, spec)
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    return {
        'name': name,
        'note': spec['note'],
        'kind': spec['kind'],
        'strength': spec['strength'],
        'include_q3s3': spec.get('include_q3s3', True),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
        'full_oof': v56.evaluate(train, oof),
        'role_oof': v56.role_evaluations(train, oof, train_roles),
        'distribution_vs_anchor': v56.describe_vs_anchor(submission, anchor_sub, test_roles),
    }


def main():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_SAMPLE_PATH)
    profiles, test_roles = v56.build_test_profiles(train, sub)
    train_roles = v56.build_train_roles(train, profiles)

    anchor_oof = v56.load_oof(ANCHOR_TAG, train)
    anchor_sub = v56.load_submission(ANCHOR_TAG, sub)
    source_oofs = {name: v56.load_oof(tag, train) for name, tag in SOURCE_TAGS.items()}
    source_subs = {name: v56.load_submission(tag, sub) for name, tag in SOURCE_TAGS.items()}

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    candidates = [
        save_candidate(name, train, sub, anchor_oof, anchor_sub, source_oofs, source_subs, train_roles, test_roles, spec)
        for name, spec in CANDIDATES.items()
    ]

    summary = {
        'exp_tag': 'v58_centered_rank_jump',
        'anchor': {
            'tag': ANCHOR_TAG,
            'known_public_score': 0.5798876532,
            'eval': anchor_eval,
        },
        'failed_parent': {
            'tag': 'v57_raw_role_hybrid_w25',
            'known_public_score': 0.5862657301,
            'diagnosis': 'Direct raw blending moved target means too far from the v56 anchor.',
        },
        'source_tags': SOURCE_TAGS,
        'role_counts': {
            'train_pseudo': train_roles.value_counts().astype(int).to_dict(),
            'test': test_roles.value_counts().astype(int).to_dict(),
        },
        'policy_notes': [
            'Centered candidates preserve each role/target mean of the anchor.',
            'Rank-remap candidates preserve each role/target anchor distribution and use only source ordering.',
            'Block-centered candidates use subject-role state shifts, not row-level raw calibration.',
        ],
        'candidates': candidates,
        'recommended_submit_order': [
            'v58_rank_remap_w100',
            'v58_centered_delta_w50',
            'v58_block_centered_w100',
            'v58_rank_remap_no_q3s3_w100',
        ],
    }
    path = SUMMARY_DIR / 'summary_v58_centered_rank_jump.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v58] summary={path}')
    anchor_routed = anchor_eval['role_oof']['routed_rows']['loss']
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        full = item['full_oof']['loss']
        mad = item['distribution_vs_anchor']['mad_vs_anchor']
        print(
            f"  {item['name']}: full_oof={full:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed={routed - anchor_routed:+.6f} "
            f"sub_mad={mad:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

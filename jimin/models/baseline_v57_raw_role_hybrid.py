from __future__ import annotations

import json
from pathlib import Path

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


def routes_for_weight(weight: float, *, include_q3s3: bool = True):
    routes = []
    for route in ROUTE_TEMPLATE:
        if not include_q3s3 and route['target'] in {'Q3', 'S3'}:
            continue
        routes.append({**route, 'weight': float(weight)})
    return routes


CANDIDATES = {
    'v57_raw_role_hybrid_w25': {
        'routes': routes_for_weight(0.25),
        'note': 'Raw role hybrid at 25% strength; first safer big-swing probe.',
    },
    'v57_raw_role_hybrid_w50': {
        'routes': routes_for_weight(0.50),
        'note': 'Raw role hybrid at 50% strength.',
    },
    'v57_raw_role_hybrid_w75': {
        'routes': routes_for_weight(0.75),
        'note': 'Raw role hybrid at 75% strength; high-risk/high-reward.',
    },
    'v57_raw_role_hybrid_w100': {
        'routes': routes_for_weight(1.00),
        'note': 'Full selected raw sources by role/target; maximum structural probe.',
    },
    'v57_raw_role_hybrid_no_q3s3_w50': {
        'routes': routes_for_weight(0.50, include_q3s3=False),
        'note': '50% raw role hybrid but freezes Q3/S3 for public no_q3s3 caution.',
    },
}


def save_candidate(name, train, sub, anchor_oof, anchor_sub, source_oofs, source_subs, train_roles, test_roles, spec):
    oof = v56.apply_router(anchor_oof, source_oofs, train_roles, spec['routes'])
    submission = v56.apply_router(anchor_sub, source_subs, test_roles, spec['routes'])
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    return {
        'name': name,
        'note': spec['note'],
        'routes': spec['routes'],
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
        'exp_tag': 'v57_raw_role_hybrid',
        'anchor': {
            'tag': ANCHOR_TAG,
            'known_public_score': 0.5798876532,
            'eval': anchor_eval,
        },
        'source_tags': SOURCE_TAGS,
        'role_counts': {
            'train_pseudo': train_roles.value_counts().astype(int).to_dict(),
            'test': test_roles.value_counts().astype(int).to_dict(),
        },
        'policy_notes': [
            'This is intentionally a big-swing family; v56-style tiny deltas cannot close the current public gap.',
            'w25 is the first recommended probe; w50/w75/w100 are escalation candidates if public responds well.',
            'no_q3s3_w50 is included because public feedback previously punished global Q3/S3 deltas.',
        ],
        'candidates': candidates,
        'recommended_submit_order': [
            'v57_raw_role_hybrid_w25',
            'v57_raw_role_hybrid_no_q3s3_w50',
            'v57_raw_role_hybrid_w50',
            'v57_raw_role_hybrid_w75',
            'v57_raw_role_hybrid_w100',
        ],
    }
    path = SUMMARY_DIR / 'summary_v57_raw_role_hybrid.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v57] summary={path}')
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

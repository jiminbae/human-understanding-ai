"""v56: conservative block-aware target router.

This is a no-training post-processing layer on top of the current public-best
anchor.  It routes only small target-wise deltas from older sources into rows
whose observed test position looks like simple interior, fragmented interior,
or final tail blocks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']
ROLE_COL = 'v56_role'

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_TAG = 'v55_no_q3s3_scale104'
SOURCE_TAGS = {
    'v28a': 'v28a_fwd_only',
    'v28b': 'v28b_pseudo85_fwd',
    'v30': 'v30_bidir_history_extended',
    'v34': 'v35_winning_policy_ablation_q1p1_q3s4p2',
    's4_boost': 'v55_no_q3s3_s4_boost115',
    'scale108': 'v55_no_q3s3_scale108',
    'v53_mid': 'v53_public_mid_probe',
}


def ensure_dirs() -> None:
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def align_to_keys(frame: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    aligned = keys.merge(frame[KEYS + TARGETS], on=KEYS, how='left', validate='one_to_one')
    if aligned[TARGETS].isna().any().any():
        missing = aligned.loc[aligned[TARGETS].isna().any(axis=1), KEYS].head()
        raise ValueError(f'Could not align prediction frame. Missing keys:\n{missing}')
    return aligned


def load_oof(tag: str, train: pd.DataFrame) -> pd.DataFrame:
    return align_to_keys(load_frame(OOF_DIR / f'oof_{tag}.csv'), train[KEYS])


def load_submission(tag: str, sub: pd.DataFrame) -> pd.DataFrame:
    return align_to_keys(load_frame(SUB_DIR / f'submission_{tag}.csv'), sub[KEYS])


def target_logloss(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(y_pred)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate(train: pd.DataFrame, pred: pd.DataFrame, mask=None):
    if mask is None:
        mask = np.ones(len(train), dtype=bool)
    per = {
        target: target_logloss(train.loc[mask, target], pred.loc[mask, target])
        for target in TARGETS
    }
    return {
        'loss': float(np.mean(list(per.values()))),
        'per_target': per,
        'n_rows': int(np.sum(mask)),
    }


def build_test_profiles(train: pd.DataFrame, sub: pd.DataFrame):
    profiles = {}
    role = pd.Series(index=sub.index, dtype=object)
    for sid in sorted(train['subject_id'].unique()):
        tr = train.loc[train['subject_id'] == sid, ['sleep_date']].assign(kind='T', row_id=-1)
        te = sub.loc[sub['subject_id'] == sid, ['sleep_date']].assign(kind='X', row_id=lambda x: x.index)
        combined = pd.concat([tr, te], ignore_index=False).sort_values('sleep_date')

        runs = []
        for row in combined.itertuples(index=False):
            if not runs or runs[-1]['kind'] != row.kind:
                runs.append({'kind': row.kind, 'row_ids': [int(row.row_id)]})
            else:
                runs[-1]['row_ids'].append(int(row.row_id))

        x_runs = [run for run in runs if run['kind'] == 'X']
        t_runs = [len(run['row_ids']) for run in runs if run['kind'] == 'T']
        x_lengths = [len(run['row_ids']) for run in x_runs]
        is_simple = len(x_runs) <= 2
        interior_role = 'simple_interior' if is_simple else 'fragmented_interior'
        for i, run in enumerate(x_runs):
            run_role = 'tail' if i == len(x_runs) - 1 else interior_role
            role.loc[run['row_ids']] = run_role
        profiles[sid] = {
            'runs': [
                {'kind': run['kind'], 'n': len(run['row_ids'])}
                for run in runs
            ],
            't_runs': t_runs,
            'x_runs': x_lengths,
            'interior_x_runs': x_lengths[:-1],
            'tail_x_run': x_lengths[-1],
            'n_x_runs': len(x_lengths),
            'is_simple': is_simple,
        }
    if role.isna().any():
        raise ValueError('Some test rows were not assigned a v56 role.')
    return profiles, role


def proportional_gaps(visible_total: int, raw_gaps):
    raw = np.asarray(raw_gaps, dtype=float)
    n_gaps = len(raw)
    if n_gaps == 0:
        return []
    if visible_total < n_gaps:
        base = np.zeros(n_gaps, dtype=int)
        base[:visible_total] = 1
        return base.tolist()

    base = np.ones(n_gaps, dtype=int)
    remaining = int(visible_total - n_gaps)
    if remaining <= 0:
        return base.tolist()

    raw = np.maximum(raw, 1.0)
    shares = remaining * raw / raw.sum()
    add = np.floor(shares).astype(int)
    leftover = remaining - int(add.sum())
    if leftover > 0:
        order = np.argsort(-(shares - add))
        add[order[:leftover]] += 1
    return (base + add).tolist()


def build_train_roles(train: pd.DataFrame, profiles) -> pd.Series:
    role = pd.Series('visible', index=train.index, dtype=object)
    for sid, grp in train.groupby('subject_id', sort=True):
        profile = profiles[sid]
        idx = grp.sort_values('sleep_date').index.to_numpy()
        x_lengths = profile['x_runs']
        hidden_total = int(sum(x_lengths))
        visible_total = len(idx) - hidden_total
        if visible_total < 0:
            raise ValueError(f'Profile hides too many rows for {sid}: {hidden_total}>{len(idx)}')
        gaps = proportional_gaps(visible_total, profile['t_runs'])
        if len(gaps) < len(x_lengths):
            raise ValueError(f'Not enough visible gaps to place hidden runs for {sid}')

        interior_role = 'simple_interior' if profile['is_simple'] else 'fragmented_interior'
        cursor = 0
        for i, x_len in enumerate(x_lengths):
            cursor += gaps[i]
            selected = idx[cursor:cursor + x_len]
            run_role = 'tail' if i == len(x_lengths) - 1 else interior_role
            role.loc[selected] = run_role
            cursor += x_len
    return role


def apply_router(anchor: pd.DataFrame, sources: dict[str, pd.DataFrame], roles: pd.Series, route_spec):
    out = anchor.copy()
    for route in route_spec:
        mask = roles == route['role']
        if not bool(mask.any()):
            continue
        source = sources[route['source']]
        weight = float(route['weight'])
        target = route['target']
        out.loc[mask, target] = clip_prob(
            (1.0 - weight) * out.loc[mask, target] + weight * source.loc[mask, target]
        )
    return out


def describe_vs_anchor(pred: pd.DataFrame, anchor: pd.DataFrame, roles: pd.Series):
    diff = pred[TARGETS] - anchor[TARGETS]
    summary = {
        'mad_vs_anchor': float(diff.abs().to_numpy().mean()),
        'max_abs_vs_anchor': float(diff.abs().to_numpy().max()),
        'per_target_mad': {target: float(diff[target].abs().mean()) for target in TARGETS},
        'mean_delta': {target: float(diff[target].mean()) for target in TARGETS},
        'role_mad': {},
    }
    for role_name in ['simple_interior', 'fragmented_interior', 'tail']:
        mask = roles == role_name
        role_diff = diff.loc[mask]
        summary['role_mad'][role_name] = {
            'n_rows': int(mask.sum()),
            'mad': float(role_diff.abs().to_numpy().mean()) if len(role_diff) else 0.0,
            'per_target_mad': {
                target: float(role_diff[target].abs().mean()) if len(role_diff) else 0.0
                for target in TARGETS
            },
        }
    return summary


def role_evaluations(train: pd.DataFrame, pred: pd.DataFrame, roles: pd.Series):
    result = {
        'routed_rows': evaluate(train, pred, roles != 'visible'),
    }
    for role_name in ['simple_interior', 'fragmented_interior', 'tail']:
        result[role_name] = evaluate(train, pred, roles == role_name)
    return result


SAFE_ROUTES = [
    {'role': 'fragmented_interior', 'target': 'Q2', 'source': 'v30', 'weight': 0.15},
    {'role': 'fragmented_interior', 'target': 'S4', 'source': 'v30', 'weight': 0.15},
    {'role': 'tail', 'target': 'S4', 'source': 's4_boost', 'weight': 1.0},
]

MID_ROUTES = [
    {'role': 'fragmented_interior', 'target': 'Q2', 'source': 'v30', 'weight': 0.30},
    {'role': 'fragmented_interior', 'target': 'S4', 'source': 'v30', 'weight': 0.25},
    {'role': 'simple_interior', 'target': 'S4', 'source': 'v28b', 'weight': 0.08},
    {'role': 'tail', 'target': 'S4', 'source': 's4_boost', 'weight': 1.0},
]

Q3S3_PROBE_ROUTES = SAFE_ROUTES + [
    {'role': 'simple_interior', 'target': 'Q3', 'source': 'v30', 'weight': 0.10},
    {'role': 'simple_interior', 'target': 'S3', 'source': 'v53_mid', 'weight': 0.10},
    {'role': 'fragmented_interior', 'target': 'Q3', 'source': 'v30', 'weight': 0.10},
    {'role': 'fragmented_interior', 'target': 'S3', 'source': 'v28b', 'weight': 0.10},
    {'role': 'tail', 'target': 'Q3', 'source': 'v53_mid', 'weight': 0.10},
    {'role': 'tail', 'target': 'S3', 'source': 'v53_mid', 'weight': 0.10},
]

SCALE108_ROUTER_ROUTES = [
    {'role': role, 'target': target, 'source': 'scale108', 'weight': 1.0}
    for role in ['simple_interior', 'fragmented_interior', 'tail']
    for target in ['Q1', 'Q2', 'S1', 'S2', 'S4']
] + SAFE_ROUTES

CANDIDATES = {
    'v56_block_router_safe': {
        'routes': SAFE_ROUTES,
        'note': 'OOF-guarded Q2/S4 router; Q3/S3 frozen and only tail S4 uses the s4_boost source.',
    },
    'v56_block_router_mid': {
        'routes': MID_ROUTES,
        'note': 'Stronger Q2/S4 router with a small simple-interior S4 probe; Q3/S3 frozen.',
    },
    'v56_block_router_q3s3_probe': {
        'routes': Q3S3_PROBE_ROUTES,
        'note': 'Riskier probe that adds tiny role-specific Q3/S3 deltas despite no_q3s3 public feedback.',
    },
    'v56_scale108_plus_router_safe': {
        'routes': SCALE108_ROUTER_ROUTES,
        'note': 'Use scale108 for core targets, then add the safe Q2/S4 router.',
    },
}


def save_candidate(name, train, sub, anchor_oof, anchor_sub, source_oofs, source_subs, train_roles, test_roles, spec):
    oof = apply_router(anchor_oof, source_oofs, train_roles, spec['routes'])
    submission = apply_router(anchor_sub, source_subs, test_roles, spec['routes'])

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
        'full_oof': evaluate(train, oof),
        'role_oof': role_evaluations(train, oof, train_roles),
        'distribution_vs_anchor': describe_vs_anchor(submission, anchor_sub, test_roles),
    }


def main():
    ensure_dirs()
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_SAMPLE_PATH)

    profiles, test_roles = build_test_profiles(train, sub)
    train_roles = build_train_roles(train, profiles)

    anchor_oof = load_oof(ANCHOR_TAG, train)
    anchor_sub = load_submission(ANCHOR_TAG, sub)
    source_oofs = {name: load_oof(tag, train) for name, tag in SOURCE_TAGS.items()}
    source_subs = {name: load_submission(tag, sub) for name, tag in SOURCE_TAGS.items()}

    anchor_eval = {
        'full_oof': evaluate(train, anchor_oof),
        'role_oof': role_evaluations(train, anchor_oof, train_roles),
    }
    candidates = [
        save_candidate(
            name,
            train,
            sub,
            anchor_oof,
            anchor_sub,
            source_oofs,
            source_subs,
            train_roles,
            test_roles,
            spec,
        )
        for name, spec in CANDIDATES.items()
    ]

    summary = {
        'exp_tag': 'v56_block_router',
        'anchor': {
            'tag': ANCHOR_TAG,
            'known_public_score': 0.5799090135,
            'oof_path': str(OOF_DIR / f'oof_{ANCHOR_TAG}.csv'),
            'submission_path': str(SUB_DIR / f'submission_{ANCHOR_TAG}.csv'),
            'eval': anchor_eval,
        },
        'source_tags': SOURCE_TAGS,
        'profiles': profiles,
        'role_counts': {
            'train_pseudo': train_roles.value_counts().astype(int).to_dict(),
            'test': test_roles.value_counts().astype(int).to_dict(),
        },
        'policy_notes': [
            'The safe/mid candidates freeze Q3 and S3 after public feedback favored no_q3s3.',
            'Tail is frozen except for the OOF-positive S4 boost direction.',
            'The q3s3_probe candidate is intentionally marked risky despite its pseudo-role OOF gain.',
            'Weights are small because prior OOF winners did not transfer cleanly to public LB.',
        ],
        'candidates': candidates,
        'recommended_submit_order': [
            'v56_block_router_safe',
            'v56_block_router_mid',
            'v56_scale108_plus_router_safe',
            'v56_block_router_q3s3_probe',
        ],
    }
    path = SUMMARY_DIR / 'summary_v56_block_router.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v56] summary={path}')
    print(f'[v56] role_counts={summary["role_counts"]}')
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

"""Evaluate historical sources under richer pseudo-interior masks.

The real public/test split contains 156 interior hidden rows and 94 final-tail
rows. Earlier diagnostics used only the first hidden block per subject as a
middle proxy, which under-represented fragmented subjects. This script rebuilds
all interior hidden blocks from the observed T/X run profiles and evaluates
historical OOF sources on:
  - simple interior blocks
  - fragmented interior blocks
  - all interior blocks together
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OOF_DIR = BASE_DIR / 'outputs' / 'oof'
OUT_DIR = BASE_DIR / 'outputs' / 'analysis'
SUMMARY_PATH = OUT_DIR / 'pseudo_public_interior_profile_eval.json'

OOF_PATHS = {
    'v28a': OOF_DIR / 'oof_v28a_fwd_only.csv',
    'v28b': OOF_DIR / 'oof_v28b_pseudo85_fwd.csv',
    'v29': OOF_DIR / 'oof_v29_bidirectional_target_history.csv',
    'v30': OOF_DIR / 'oof_v30_bidir_history_extended.csv',
    'v32': OOF_DIR / 'oof_v32_target_specialized_bidir_s2w0p675.csv',
    'v34': OOF_DIR / 'oof_v35_winning_policy_ablation_q1p1_q3s4p2.csv',
}


def load_frames():
    train = pd.read_csv(TRAIN_PATH, parse_dates=['sleep_date', 'lifelog_date'])
    sub = pd.read_csv(SUB_PATH, parse_dates=['sleep_date', 'lifelog_date'])
    oofs = {
        name: pd.read_csv(path, parse_dates=['sleep_date', 'lifelog_date'])
        for name, path in OOF_PATHS.items()
    }
    return train.reset_index(drop=True), sub.reset_index(drop=True), oofs


def build_profiles(train, sub):
    profiles = {}
    for sid in sorted(train['subject_id'].unique()):
        combined = pd.concat([
            train.loc[train['subject_id'] == sid, ['sleep_date']].assign(kind='T'),
            sub.loc[sub['subject_id'] == sid, ['sleep_date']].assign(kind='X'),
        ]).sort_values('sleep_date')

        runs = []
        for row in combined.itertuples(index=False):
            if not runs or runs[-1]['kind'] != row.kind:
                runs.append({'kind': row.kind, 'n': 1})
            else:
                runs[-1]['n'] += 1

        t_runs = [run['n'] for run in runs if run['kind'] == 'T']
        x_runs = [run['n'] for run in runs if run['kind'] == 'X']
        profiles[sid] = {
            'runs': runs,
            't_runs': t_runs,
            'x_runs': x_runs,
            'interior_x_runs': x_runs[:-1],
            'tail_x_run': x_runs[-1],
            'n_x_runs': len(x_runs),
            'is_simple': len(x_runs) <= 2,
            'is_fragmented': len(x_runs) >= 5,
        }
    return profiles


def proportional_gaps(visible_total, raw_gaps):
    raw = np.asarray(raw_gaps, dtype=float)
    n_gaps = len(raw)
    if visible_total < n_gaps:
        raise ValueError('visible_total must leave at least one row per gap')

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


def interior_mask_for_subject(train_grp, profile):
    idx = train_grp.sort_values('sleep_date').index.to_numpy()
    lengths = list(profile['interior_x_runs'])
    if not lengths:
        return []

    hidden_total = int(sum(lengths))
    visible_total = len(idx) - hidden_total
    raw_gaps = profile['t_runs'][:len(lengths) + 1]
    gaps = proportional_gaps(visible_total, raw_gaps)

    selected = []
    cursor = gaps[0]
    for block_i, block_len in enumerate(lengths):
        selected.extend(idx[cursor:cursor + block_len].tolist())
        cursor += block_len
        cursor += gaps[block_i + 1]
    return selected


def build_interior_masks(train, profiles):
    simple = pd.Series(False, index=train.index)
    fragmented = pd.Series(False, index=train.index)
    all_interior = pd.Series(False, index=train.index)

    for sid, grp in train.groupby('subject_id', sort=True):
        selected = interior_mask_for_subject(grp, profiles[sid])
        if not selected:
            continue
        all_interior.loc[selected] = True
        if profiles[sid]['is_simple']:
            simple.loc[selected] = True
        if profiles[sid]['is_fragmented']:
            fragmented.loc[selected] = True
    return simple, fragmented, all_interior


def mask_summary(train, mask):
    return {
        'n_hidden': int(mask.sum()),
        'per_subject_hidden': (
            train.loc[mask]
            .groupby('subject_id')
            .size()
            .reindex(sorted(train['subject_id'].unique()), fill_value=0)
            .astype(int)
            .to_dict()
        ),
    }


def evaluate_source(train, pred, mask):
    per_target = {
        target: float(log_loss(
            train.loc[mask, target].values,
            np.clip(pred.loc[mask, target].values, 1e-7, 1 - 1e-7),
        ))
        for target in TARGETS
    }
    return {
        'loss': float(np.mean(list(per_target.values()))),
        'per_target': per_target,
    }


def rank_sources_by_target(results, split_name):
    ranked = {}
    split_results = results[split_name]
    for target in TARGETS:
        ranked[target] = sorted(
            [
                {
                    'source': source,
                    'loss': metrics['per_target'][target],
                }
                for source, metrics in split_results.items()
            ],
            key=lambda row: row['loss'],
        )
    return ranked


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, sub, oofs = load_frames()
    profiles = build_profiles(train, sub)
    simple_mask, fragmented_mask, all_mask = build_interior_masks(train, profiles)

    masks = {
        'simple_interior': simple_mask,
        'fragmented_interior': fragmented_mask,
        'all_interior': all_mask,
    }
    results = {
        split_name: {
            source: evaluate_source(train, pred, mask)
            for source, pred in oofs.items()
        }
        for split_name, mask in masks.items()
    }
    rankings = {
        split_name: rank_sources_by_target(results, split_name)
        for split_name in masks
    }

    summary = {
        'profiles': profiles,
        'mask_summaries': {
            split_name: mask_summary(train, mask)
            for split_name, mask in masks.items()
        },
        'results': results,
        'rankings': rankings,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    for split_name in masks:
        print(f'\n[{split_name}] n={int(masks[split_name].sum())}')
        for source, metrics in sorted(results[split_name].items(), key=lambda item: item[1]['loss']):
            per = {k: round(v, 6) for k, v in metrics['per_target'].items()}
            print(f'  {source}: {metrics["loss"]:.6f} {per}')
    print(f'\nsummary saved: {SUMMARY_PATH}')


if __name__ == '__main__':
    main()

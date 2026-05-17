"""Evaluate label-only reconstruction under pseudo-public temporal masks.

The real test split is not a simple future holdout. Most subjects alternate between
visible train blocks and hidden test blocks. This script hides train labels in
similar middle/tail patterns and asks how much of the target signal can be
recovered from the remaining labels alone.
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
TEST_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUT_DIR = BASE_DIR / 'outputs' / 'analysis'
SUMMARY_PATH = OUT_DIR / 'pseudo_public_temporal_mask_eval.json'


def load_frames():
    train = pd.read_csv(TRAIN_PATH, parse_dates=['sleep_date', 'lifelog_date'])
    test = pd.read_csv(TEST_PATH, parse_dates=['sleep_date', 'lifelog_date'])
    return train.sort_values(['subject_id', 'sleep_date']).reset_index(drop=True), test


def build_test_run_profile(train: pd.DataFrame, test: pd.DataFrame):
    profiles = {}
    for sid in sorted(train['subject_id'].unique()):
        combined = pd.concat([
            train.loc[train['subject_id'] == sid, ['sleep_date']].assign(kind='T'),
            test.loc[test['subject_id'] == sid, ['sleep_date']].assign(kind='X'),
        ]).sort_values('sleep_date')

        runs = []
        for row in combined.itertuples(index=False):
            if not runs or runs[-1]['kind'] != row.kind:
                runs.append({'kind': row.kind, 'n': 1})
            else:
                runs[-1]['n'] += 1
        x_runs = [run['n'] for run in runs if run['kind'] == 'X']
        profiles[sid] = {
            'runs': runs,
            'x_runs': x_runs,
            'first_x': x_runs[0],
            'last_x': x_runs[-1],
            'x_total': int(sum(x_runs)),
        }
    return profiles


def empty_mask(train: pd.DataFrame):
    return pd.Series(False, index=train.index)


def subject_positions(train: pd.DataFrame):
    return {
        sid: grp.index.to_numpy()
        for sid, grp in train.groupby('subject_id', sort=True)
    }


def make_middle_mask(train: pd.DataFrame, profiles, min_visible_each_side=6):
    mask = empty_mask(train)
    for sid, idx in subject_positions(train).items():
        n = len(idx)
        length = min(profiles[sid]['first_x'], max(0, n - 2 * min_visible_each_side))
        if length <= 0:
            continue
        start = int(np.clip((n - length) // 2, min_visible_each_side, n - length - min_visible_each_side))
        mask.loc[idx[start:start + length]] = True
    return mask


def make_tail_mask(train: pd.DataFrame, profiles, min_visible=8):
    mask = empty_mask(train)
    for sid, idx in subject_positions(train).items():
        n = len(idx)
        length = min(profiles[sid]['last_x'], max(0, n - min_visible))
        if length <= 0:
            continue
        mask.loc[idx[-length:]] = True
    return mask


def make_hybrid_mask(train: pd.DataFrame, profiles, min_visible_each_side=6, min_visible_total=10):
    mask = empty_mask(train)
    for sid, idx in subject_positions(train).items():
        n = len(idx)
        middle_len = profiles[sid]['first_x']
        tail_len = profiles[sid]['last_x']
        max_hidden = max(0, n - min_visible_total)
        if max_hidden <= 0:
            continue

        if middle_len + tail_len > max_hidden:
            scale = max_hidden / max(1, middle_len + tail_len)
            middle_len = int(np.floor(middle_len * scale))
            tail_len = int(np.floor(tail_len * scale))
            if middle_len + tail_len == 0:
                continue

        tail_start = n - tail_len
        max_middle_len = max(0, tail_start - 2 * min_visible_each_side)
        middle_len = min(middle_len, max_middle_len)
        if middle_len > 0:
            middle_start = int(np.clip(
                (tail_start - middle_len) // 2,
                min_visible_each_side,
                tail_start - middle_len - min_visible_each_side,
            ))
            mask.loc[idx[middle_start:middle_start + middle_len]] = True
        if tail_len > 0:
            mask.loc[idx[tail_start:]] = True
    return mask


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def subject_prior(visible_values, global_prior):
    return float(np.mean(visible_values)) if len(visible_values) else float(global_prior)


def get_neighbors(visible_dates, visible_values, query_date):
    deltas = np.array([(query_date - d).days for d in visible_dates], dtype=float)
    past = np.where(deltas > 0)[0]
    future = np.where(deltas < 0)[0]
    past_idx = past[np.argmin(deltas[past])] if len(past) else None
    future_idx = future[np.argmax(deltas[future])] if len(future) else None
    return past_idx, future_idx


def reconstruct_target(train, mask, target, method):
    global_prior = float(train.loc[~mask, target].mean())
    pred = np.full(len(train), np.nan, dtype=float)

    for sid, grp in train.groupby('subject_id', sort=True):
        hidden = grp[mask.loc[grp.index]]
        visible = grp[~mask.loc[grp.index]]
        visible_dates = visible['sleep_date'].tolist()
        visible_values = visible[target].to_numpy(dtype=float)
        subj_prior = subject_prior(visible_values, global_prior)

        for row in hidden.itertuples():
            past_idx, future_idx = get_neighbors(visible_dates, visible_values, row.sleep_date)

            if method == 'global_mean':
                value = global_prior
            elif method == 'subject_mean':
                value = subj_prior
            elif method == 'nearest_past':
                value = visible_values[past_idx] if past_idx is not None else subj_prior
            elif method == 'nearest_future':
                value = visible_values[future_idx] if future_idx is not None else subj_prior
            elif method == 'nearest_any':
                if past_idx is None and future_idx is None:
                    value = subj_prior
                elif past_idx is None:
                    value = visible_values[future_idx]
                elif future_idx is None:
                    value = visible_values[past_idx]
                else:
                    past_gap = abs((row.sleep_date - visible_dates[past_idx]).days)
                    future_gap = abs((visible_dates[future_idx] - row.sleep_date).days)
                    value = visible_values[past_idx] if past_gap <= future_gap else visible_values[future_idx]
            elif method == 'linear_interp':
                if past_idx is None or future_idx is None:
                    value = subj_prior
                else:
                    past_gap = abs((row.sleep_date - visible_dates[past_idx]).days)
                    future_gap = abs((visible_dates[future_idx] - row.sleep_date).days)
                    denom = max(1, past_gap + future_gap)
                    value = (
                        future_gap * visible_values[past_idx]
                        + past_gap * visible_values[future_idx]
                    ) / denom
            elif method == 'invdist_bidir':
                if past_idx is None and future_idx is None:
                    value = subj_prior
                elif past_idx is None:
                    value = visible_values[future_idx]
                elif future_idx is None:
                    value = visible_values[past_idx]
                else:
                    past_gap = max(1, abs((row.sleep_date - visible_dates[past_idx]).days))
                    future_gap = max(1, abs((visible_dates[future_idx] - row.sleep_date).days))
                    w_past = 1.0 / past_gap
                    w_future = 1.0 / future_gap
                    value = (
                        w_past * visible_values[past_idx]
                        + w_future * visible_values[future_idx]
                    ) / (w_past + w_future)
            elif method.startswith('exp_kernel_'):
                bandwidth = float(method.rsplit('_', 1)[-1])
                if not len(visible_values):
                    value = global_prior
                else:
                    gaps = np.array([abs((row.sleep_date - d).days) for d in visible_dates], dtype=float)
                    weights = np.exp(-gaps / bandwidth)
                    value = float(np.dot(weights, visible_values) / np.maximum(weights.sum(), 1e-9))
            else:
                raise ValueError(f'Unknown method: {method}')

            pred[row.Index] = value
    return clip_prob(pred[mask.to_numpy()])


def evaluate_methods(train, mask, split_name):
    methods = [
        'global_mean',
        'subject_mean',
        'nearest_past',
        'nearest_future',
        'nearest_any',
        'linear_interp',
        'invdist_bidir',
        'exp_kernel_3',
        'exp_kernel_7',
        'exp_kernel_14',
        'exp_kernel_30',
    ]
    y = train.loc[mask, TARGETS].reset_index(drop=True)
    rows = []
    for method in methods:
        per_target = {}
        for target in TARGETS:
            pred = reconstruct_target(train, mask, target, method)
            per_target[target] = float(log_loss(y[target], pred))
        rows.append({
            'split': split_name,
            'method': method,
            'loss': float(np.mean(list(per_target.values()))),
            'per_target': per_target,
        })
    return rows


def mask_summary(train, mask):
    per_subject = (
        train.loc[mask]
        .groupby('subject_id')
        .size()
        .reindex(sorted(train['subject_id'].unique()), fill_value=0)
        .astype(int)
        .to_dict()
    )
    return {
        'n_hidden': int(mask.sum()),
        'per_subject_hidden': per_subject,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_frames()
    profiles = build_test_run_profile(train, test)
    masks = {
        'middle_block': make_middle_mask(train, profiles),
        'tail_block': make_tail_mask(train, profiles),
        'hybrid_middle_tail': make_hybrid_mask(train, profiles),
    }

    results = []
    for split_name, mask in masks.items():
        results.extend(evaluate_methods(train, mask, split_name))

    payload = {
        'test_run_profiles': profiles,
        'mask_summaries': {name: mask_summary(train, mask) for name, mask in masks.items()},
        'results': results,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')

    print('Pseudo-public temporal mask evaluation')
    print(f'summary saved: {SUMMARY_PATH}')
    for split_name in masks:
        print(f'\n[{split_name}] {mask_summary(train, masks[split_name])}')
        split_rows = [row for row in results if row['split'] == split_name]
        for row in sorted(split_rows, key=lambda x: x['loss']):
            print(f"  {row['method']:<16} loss={row['loss']:.6f}  per_target={row['per_target']}")


if __name__ == '__main__':
    main()

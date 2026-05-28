from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
RAW_DIR = BASE_DIR / 'ch2025_data_items'
OUT_DIR = BASE_DIR / 'outputs' / 'analysis'
OUT_PATH = OUT_DIR / 'data_audit_next_direction.json'


def load_keys():
    train = pd.read_csv(TRAIN_PATH, parse_dates=['sleep_date', 'lifelog_date'])
    sub = pd.read_csv(SUB_PATH, parse_dates=['sleep_date', 'lifelog_date'])
    return train.reset_index(drop=True), sub.reset_index(drop=True)


def runs_for_subject(train, sub, sid):
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
    return runs


def key_summary(train, sub):
    rows = []
    for sid in sorted(train['subject_id'].unique()):
        tr = train[train['subject_id'] == sid].sort_values('sleep_date')
        te = sub[sub['subject_id'] == sid].sort_values('sleep_date')
        train_gaps = tr['sleep_date'].diff().dt.days.dropna()
        test_gaps = te['sleep_date'].diff().dt.days.dropna()
        runs = runs_for_subject(train, sub, sid)
        rows.append({
            'subject_id': sid,
            'train_rows': int(len(tr)),
            'test_rows': int(len(te)),
            'train_sleep_min': str(tr['sleep_date'].min().date()),
            'train_sleep_max': str(tr['sleep_date'].max().date()),
            'test_sleep_min': str(te['sleep_date'].min().date()),
            'test_sleep_max': str(te['sleep_date'].max().date()),
            'train_gap_max_days': int(train_gaps.max()) if len(train_gaps) else 0,
            'test_gap_max_days': int(test_gaps.max()) if len(test_gaps) else 0,
            'runs': runs,
            'x_runs': [r['n'] for r in runs if r['kind'] == 'X'],
        })
    return rows


def target_summary(train):
    overall = {}
    by_subject = {}
    for target in TARGETS:
        p = float(train[target].mean())
        overall[target] = {
            'positive_rate': p,
            'count_1': int(train[target].sum()),
            'count_0': int((1 - train[target]).sum()),
        }
    for sid, grp in train.groupby('subject_id', sort=True):
        by_subject[sid] = {
            target: float(grp[target].mean())
            for target in TARGETS
        }
    corr = train[TARGETS].corr().round(4).to_dict()
    return {'overall': overall, 'by_subject': by_subject, 'corr': corr}


def raw_sample_stats(path):
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    date = df['timestamp'].dt.normalize()
    per_subject = df.groupby('subject_id').size()
    per_day = df.assign(date=date).groupby(['subject_id', 'date']).size()
    value_col = [c for c in df.columns if c not in ['subject_id', 'timestamp']]
    samples = {}
    for col in value_col:
        vals = df[col].dropna().astype(str)
        samples[col] = {
            'nunique_head100k': int(vals.head(100000).nunique()),
            'examples': vals.drop_duplicates().head(5).tolist(),
        }
    return {
        'rows': int(len(df)),
        'subjects': sorted(df['subject_id'].dropna().unique().tolist()),
        'timestamp_min': str(df['timestamp'].min()),
        'timestamp_max': str(df['timestamp'].max()),
        'unique_subject_days': int(per_day.shape[0]),
        'rows_per_subject_min': int(per_subject.min()) if len(per_subject) else 0,
        'rows_per_subject_max': int(per_subject.max()) if len(per_subject) else 0,
        'rows_per_subject_mean': float(per_subject.mean()) if len(per_subject) else 0.0,
        'rows_per_subject_day_quantiles': (
            np.quantile(per_day.to_numpy(dtype=float), [0, 0.25, 0.5, 0.75, 0.95, 1]).round(3).tolist()
            if len(per_day) else []
        ),
        'value_columns': value_col,
        'samples': samples,
    }


def raw_coverage(train, sub, raw_paths):
    key_days = pd.concat([
        train[['subject_id', 'lifelog_date']].assign(split='train'),
        sub[['subject_id', 'lifelog_date']].assign(split='test'),
    ], ignore_index=True)
    key_days['date'] = key_days['lifelog_date'].dt.normalize()
    coverage = {}
    for path in raw_paths:
        df = pd.read_parquet(path, columns=['subject_id', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        raw_days = (
            df.assign(date=df['timestamp'].dt.normalize())
            .groupby(['subject_id', 'date'])
            .size()
            .rename('raw_rows')
            .reset_index()
        )
        merged = key_days.merge(raw_days, how='left', on=['subject_id', 'date'])
        merged['has_raw'] = merged['raw_rows'].fillna(0) > 0
        split_stats = {}
        for split, grp in merged.groupby('split', sort=True):
            rows = grp['raw_rows'].fillna(0)
            split_stats[split] = {
                'key_days': int(len(grp)),
                'covered_days': int(grp['has_raw'].sum()),
                'coverage_rate': float(grp['has_raw'].mean()),
                'rows_per_key_day_nonzero_quantiles': (
                    np.quantile(rows[rows > 0], [0, 0.25, 0.5, 0.75, 0.95, 1]).round(3).tolist()
                    if (rows > 0).any() else []
                ),
            }
        coverage[path.name] = split_stats
    return coverage


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, sub = load_keys()
    raw_paths = sorted(RAW_DIR.glob('*.parquet'))
    audit = {
        'train_shape': list(train.shape),
        'test_shape': list(sub.shape),
        'key_summary': key_summary(train, sub),
        'target_summary': target_summary(train),
        'raw_files': {path.name: raw_sample_stats(path) for path in raw_paths},
        'raw_key_day_coverage': raw_coverage(train, sub, raw_paths),
    }
    OUT_PATH.write_text(json.dumps(audit, indent=2), encoding='utf-8')

    print(f'saved={OUT_PATH}')
    print(f'train_shape={audit["train_shape"]} test_shape={audit["test_shape"]}')
    print('target positive rates')
    for target, info in audit['target_summary']['overall'].items():
        print(f'  {target}: {info["positive_rate"]:.3f} ({info["count_1"]}/{info["count_0"]})')
    print('raw coverage train/test')
    for name, split_stats in audit['raw_key_day_coverage'].items():
        tr = split_stats['train']['coverage_rate']
        te = split_stats['test']['coverage_rate']
        print(f'  {name}: train={tr:.3f} test={te:.3f}')


if __name__ == '__main__':
    main()

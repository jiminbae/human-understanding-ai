# v47: hour-grid raw sensor subject-state residual model.
#   - v45/v46 found a strong split-aware S4 post-processing axis, but that is a
#     narrow correction. This script looks for a new base-signal axis by
#     expanding raw sensors into hour/segment rhythm features.
#   - The model is trained as a residual candidate against the current best
#     anchor, not as an unconditional replacement. It writes raw predictions and
#     conservative blends so public submissions can test whether raw hour-grid
#     features add signal beyond the v45 anchor.
import gc
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jimin.analysis import pseudo_public_interior_profile_eval as interior_eval
from jimin.models import baseline_v29_bidirectional_target_history as v29
from jimin.models import baseline_v45_uncertainty_temporal_smoothing as v45


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
EXP_TAG = os.environ.get('V47_EXP_TAG', 'v47_hourgrid_subject_state_residual')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'ch2025_data_items'
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
FEATURE_DIR = OUTPUTS_DIR / 'features'

DEFAULT_ANCHOR_OOF = OOF_DIR / 'oof_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
DEFAULT_ANCHOR_SUB = SUB_DIR / 'submission_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
FALLBACK_ANCHOR_OOF = OOF_DIR / 'oof_v38_block_role_aware_tail_conservative_w40.csv'
FALLBACK_ANCHOR_SUB = SUB_DIR / 'submission_v38_block_role_aware_tail_conservative_w40.csv'

N_SEEDS = int(os.environ.get('V47_N_SEEDS', '3'))
SEED_POOL = [42, 2025, 777, 1234, 314, 9999, 7, 1337]
N_ESTIMATORS = int(os.environ.get('V47_N_ESTIMATORS', '850'))
MAX_MODEL_FEATURES = int(os.environ.get('V47_MAX_MODEL_FEATURES', '280'))
MAX_STATE_COLS = int(os.environ.get('V47_MAX_STATE_COLS', '170'))
MAX_ROLL_COLS = int(os.environ.get('V47_MAX_ROLL_COLS', '90'))
USE_BASE_V29_FEATURES = os.environ.get('V47_USE_BASE_V29_FEATURES', '1') == '1'
CACHE_FEATURES = os.environ.get('V47_CACHE_FEATURES', '1') == '1'
FEATURE_CACHE_PATH = FEATURE_DIR / f'features_{EXP_TAG}.pkl'


SEGMENTS = {
    'late': [0, 1, 2],
    'sleep': list(range(0, 9)),
    'morn': list(range(6, 12)),
    'aftn': list(range(12, 18)),
    'eve': list(range(18, 22)),
    'presleep': [22, 23],
    'night': [22, 23, 0, 1, 2, 3, 4, 5],
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, LOG_DIR, SUMMARY_DIR, FEATURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['lifelog_date', 'sleep_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.reset_index(drop=True)


def load_parquet(name, columns=None):
    df = pd.read_parquet(DATA_DIR / f'ch2025_{name}.parquet', columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.normalize()
    df['hour'] = df['timestamp'].dt.hour.astype(int)
    return df


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def safe_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if len(arr) else np.nan


def to_records(value):
    if value is None:
        return []
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return value if isinstance(value, list) else []


def key_columns(train, sub):
    day_keys = pd.concat([
        train[['subject_id', 'lifelog_date']],
        sub[['subject_id', 'lifelog_date']],
    ], ignore_index=True).drop_duplicates().reset_index(drop=True)
    sleep_keys = pd.concat([
        train[['subject_id', 'sleep_date']],
        sub[['subject_id', 'sleep_date']],
    ], ignore_index=True).drop_duplicates().reset_index(drop=True)
    return day_keys, sleep_keys


def segment_mask(hours, segment):
    return np.isin(hours, SEGMENTS[segment])


def add_hour_ratios(row, prefix, hours, values, suffix):
    values = np.asarray(values, dtype=float)
    hours = np.asarray(hours, dtype=int)
    for hour in range(24):
        mask = hours == hour
        row[f'v47_{prefix}_h{hour:02d}_{suffix}'] = safe_mean(values[mask]) if mask.any() else np.nan
    for seg in SEGMENTS:
        mask = segment_mask(hours, seg)
        row[f'v47_{prefix}_{seg}_{suffix}'] = safe_mean(values[mask]) if mask.any() else np.nan


def add_hour_sums(row, prefix, hours, values, suffix):
    values = np.asarray(values, dtype=float)
    hours = np.asarray(hours, dtype=int)
    for hour in range(24):
        mask = hours == hour
        row[f'v47_{prefix}_h{hour:02d}_{suffix}'] = float(np.nansum(values[mask])) if mask.any() else np.nan
    for seg in SEGMENTS:
        mask = segment_mask(hours, seg)
        row[f'v47_{prefix}_{seg}_{suffix}'] = float(np.nansum(values[mask])) if mask.any() else np.nan


def add_segment_stats(row, prefix, hours, values, suffix):
    values = np.asarray(values, dtype=float)
    hours = np.asarray(hours, dtype=int)
    row[f'v47_{prefix}_all_{suffix}_mean'] = safe_mean(values)
    row[f'v47_{prefix}_all_{suffix}_std'] = safe_std(values)
    for seg in SEGMENTS:
        mask = segment_mask(hours, seg)
        row[f'v47_{prefix}_{seg}_{suffix}_mean'] = safe_mean(values[mask]) if mask.any() else np.nan
        row[f'v47_{prefix}_{seg}_{suffix}_std'] = safe_std(values[mask]) if mask.any() else np.nan


def extract_activity_grid():
    print('[v47] activity hour-grid...')
    df = load_parquet('mActivity', columns=['subject_id', 'timestamp', 'm_activity'])
    rows = []
    for (sid, date), grp in df.groupby(['subject_id', 'date'], sort=False):
        row = {'subject_id': sid, 'date': date}
        hours = grp['hour'].to_numpy()
        acts = pd.to_numeric(grp['m_activity'], errors='coerce').to_numpy()
        add_hour_ratios(row, 'act', hours, acts == 0, 'still')
        add_hour_ratios(row, 'act', hours, np.isin(acts, [3, 7, 8]), 'active')
        add_hour_ratios(row, 'act', hours, acts == 4, 'vehicle')
        row['v47_act_records'] = len(grp)
        rows.append(row)
    del df
    gc.collect()
    return pd.DataFrame(rows)


def extract_binary_grid(sensor_name, value_col, prefix):
    print(f'[v47] {prefix} hour-grid...')
    df = load_parquet(sensor_name, columns=['subject_id', 'timestamp', value_col])
    rows = []
    for (sid, date), grp in df.groupby(['subject_id', 'date'], sort=False):
        row = {'subject_id': sid, 'date': date}
        hours = grp['hour'].to_numpy()
        vals = pd.to_numeric(grp[value_col], errors='coerce').fillna(0).to_numpy()
        is_on = vals > 0
        add_hour_ratios(row, prefix, hours, is_on, 'ratio')
        row[f'v47_{prefix}_total_on'] = int(is_on.sum())
        row[f'v47_{prefix}_records'] = len(grp)
        if len(is_on) > 1:
            row[f'v47_{prefix}_rise_count'] = int(((~is_on[:-1]) & is_on[1:]).sum())
            row[f'v47_{prefix}_fall_count'] = int((is_on[:-1] & (~is_on[1:])).sum())
        else:
            row[f'v47_{prefix}_rise_count'] = 0
            row[f'v47_{prefix}_fall_count'] = 0
        rows.append(row)
    del df
    gc.collect()
    return pd.DataFrame(rows)


def extract_pedo_grid():
    print('[v47] pedometer hour-grid...')
    cols = ['subject_id', 'timestamp', 'step', 'distance', 'speed', 'burned_calories', 'step_frequency']
    df = load_parquet('wPedo', columns=cols)
    rows = []
    for (sid, date), grp in df.groupby(['subject_id', 'date'], sort=False):
        row = {'subject_id': sid, 'date': date}
        hours = grp['hour'].to_numpy()
        step = pd.to_numeric(grp['step'], errors='coerce').fillna(0).to_numpy()
        distance = pd.to_numeric(grp['distance'], errors='coerce').fillna(0).to_numpy()
        speed = pd.to_numeric(grp['speed'], errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy()
        calories = pd.to_numeric(grp['burned_calories'], errors='coerce').fillna(0).to_numpy()
        add_hour_sums(row, 'pedo', hours, step, 'steps')
        add_hour_sums(row, 'pedo', hours, distance, 'distance')
        add_hour_sums(row, 'pedo', hours, calories, 'calories')
        add_segment_stats(row, 'pedo', hours, speed, 'speed')
        add_hour_ratios(row, 'pedo', hours, step > 5, 'active_min')
        row['v47_pedo_total_steps'] = float(np.nansum(step))
        row['v47_pedo_total_distance'] = float(np.nansum(distance))
        row['v47_pedo_total_calories'] = float(np.nansum(calories))
        row['v47_pedo_records'] = len(grp)
        rows.append(row)
    del df
    gc.collect()
    return pd.DataFrame(rows)


def extract_light_grid(sensor_name, value_col, prefix):
    print(f'[v47] {prefix} light hour-grid...')
    df = load_parquet(sensor_name, columns=['subject_id', 'timestamp', value_col])
    rows = []
    for (sid, date), grp in df.groupby(['subject_id', 'date'], sort=False):
        row = {'subject_id': sid, 'date': date}
        hours = grp['hour'].to_numpy()
        vals = pd.to_numeric(grp[value_col], errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy()
        add_segment_stats(row, prefix, hours, vals, 'light')
        add_hour_ratios(row, prefix, hours, vals < 10, 'dark')
        add_hour_ratios(row, prefix, hours, vals > 1000, 'bright')
        row[f'v47_{prefix}_records'] = len(grp)
        rows.append(row)
    del df
    gc.collect()
    return pd.DataFrame(rows)


def extract_hr_grid():
    print('[v47] heart-rate segment/hour-grid...')
    df = load_parquet('wHr', columns=['subject_id', 'timestamp', 'heart_rate'])
    selected_hours = set(range(0, 9)) | set(range(18, 24))
    rows = []
    for (sid, date), grp in df.groupby(['subject_id', 'date'], sort=False):
        row = {'subject_id': sid, 'date': date}
        by_hour = {hour: [] for hour in range(24)}
        for hour, value in zip(grp['hour'], grp['heart_rate']):
            vals = np.asarray(to_records(value), dtype=float).ravel()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                by_hour[int(hour)].extend(vals.tolist())

        all_vals = []
        for values in by_hour.values():
            all_vals.extend(values)
        all_vals = np.asarray(all_vals, dtype=float)
        row['v47_hr_all_mean'] = safe_mean(all_vals)
        row['v47_hr_all_std'] = safe_std(all_vals)
        row['v47_hr_all_low_ratio'] = safe_mean(all_vals < 60) if len(all_vals) else np.nan
        row['v47_hr_all_high_ratio'] = safe_mean(all_vals > 95) if len(all_vals) else np.nan
        row['v47_hr_records'] = len(grp)

        for hour in sorted(selected_hours):
            vals = np.asarray(by_hour[hour], dtype=float)
            row[f'v47_hr_h{hour:02d}_mean'] = safe_mean(vals)
            row[f'v47_hr_h{hour:02d}_std'] = safe_std(vals)
            row[f'v47_hr_h{hour:02d}_high_ratio'] = safe_mean(vals > 95) if len(vals) else np.nan

        for seg, hours in SEGMENTS.items():
            vals = []
            for hour in hours:
                vals.extend(by_hour[hour])
            vals = np.asarray(vals, dtype=float)
            row[f'v47_hr_{seg}_mean'] = safe_mean(vals)
            row[f'v47_hr_{seg}_std'] = safe_std(vals)
            row[f'v47_hr_{seg}_low_ratio'] = safe_mean(vals < 60) if len(vals) else np.nan
            row[f'v47_hr_{seg}_high_ratio'] = safe_mean(vals > 95) if len(vals) else np.nan
        rows.append(row)
    del df
    gc.collect()
    return pd.DataFrame(rows)


def build_hourgrid_feature_table(train, sub):
    day_keys, sleep_keys = key_columns(train, sub)
    extractors = [
        extract_activity_grid,
        lambda: extract_binary_grid('mScreenStatus', 'm_screen_use', 'screen'),
        lambda: extract_binary_grid('mACStatus', 'm_charging', 'charge'),
        extract_pedo_grid,
        lambda: extract_light_grid('mLight', 'm_light', 'mlight'),
        lambda: extract_light_grid('wLight', 'w_light', 'wlight'),
        extract_hr_grid,
    ]

    feature_parts = []
    for fn in extractors:
        part = fn()
        feature_parts.append(part)

    daily = feature_parts[0]
    for part in feature_parts[1:]:
        daily = daily.merge(part, on=['subject_id', 'date'], how='outer')

    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values(['subject_id', 'date']).reset_index(drop=True)

    daily_feature_cols = [c for c in daily.columns if c not in ['subject_id', 'date']]
    rolling_base_cols = choose_columns_for_state_or_roll(daily, daily_feature_cols, MAX_ROLL_COLS)
    for col in rolling_base_cols:
        g = daily.groupby('subject_id')[col]
        daily[f'{col}_lag1'] = g.shift(1)
        daily[f'{col}_diff1'] = daily[col] - daily[f'{col}_lag1']
        daily[f'{col}_roll3'] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        daily[f'{col}_roll7'] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    day_features = daily.rename(columns={'date': 'lifelog_date'})
    train_day = train[['subject_id', 'lifelog_date']].merge(day_features, on=['subject_id', 'lifelog_date'], how='left')
    test_day = sub[['subject_id', 'lifelog_date']].merge(day_features, on=['subject_id', 'lifelog_date'], how='left')

    sleep_cols = []
    for col in day_features.columns:
        if col in ['subject_id', 'lifelog_date']:
            continue
        if any(token in col for token in ['_h00_', '_h01_', '_h02_', '_h03_', '_h04_', '_h05_', '_h06_', '_h07_', '_h08_', '_sleep_', '_late_']):
            sleep_cols.append(col)
    sleep_features = day_features[['subject_id', 'lifelog_date'] + sleep_cols].copy()
    sleep_features = sleep_features.rename(columns={'lifelog_date': 'sleep_date'})
    sleep_features = sleep_features.rename(columns={col: f'slp_{col}' for col in sleep_cols})
    train_sleep = train[['subject_id', 'sleep_date']].merge(sleep_features, on=['subject_id', 'sleep_date'], how='left')
    test_sleep = sub[['subject_id', 'sleep_date']].merge(sleep_features, on=['subject_id', 'sleep_date'], how='left')

    train_extra = pd.concat([
        train_day.drop(columns=['subject_id', 'lifelog_date']).reset_index(drop=True),
        train_sleep.drop(columns=['subject_id', 'sleep_date']).reset_index(drop=True),
    ], axis=1)
    test_extra = pd.concat([
        test_day.drop(columns=['subject_id', 'lifelog_date']).reset_index(drop=True),
        test_sleep.drop(columns=['subject_id', 'sleep_date']).reset_index(drop=True),
    ], axis=1)

    train_extra.columns = [f'v47x_{col}' if not col.startswith('v47_') and not col.startswith('slp_') else col for col in train_extra.columns]
    test_extra.columns = train_extra.columns
    print(f'[v47] hour-grid feature columns: {train_extra.shape[1]}')
    return train_extra, test_extra


def choose_columns_for_state_or_roll(frame, feature_cols, max_cols):
    candidates = []
    for col in feature_cols:
        vals = pd.to_numeric(frame[col], errors='coerce')
        nonnull = float(vals.notna().mean())
        if nonnull < 0.45:
            continue
        var = float(vals.var(skipna=True))
        if not np.isfinite(var) or var <= 1e-12:
            continue
        score = nonnull * np.log1p(var)
        candidates.append((score, col))
    candidates = sorted(candidates, reverse=True)
    return [col for _, col in candidates[:max_cols]]


def add_subject_state_features(train_full, test_full, feature_cols):
    combined = pd.concat([
        train_full[['subject_id'] + feature_cols].assign(_split='train'),
        test_full[['subject_id'] + feature_cols].assign(_split='test'),
    ], ignore_index=True)
    selected = choose_columns_for_state_or_roll(combined, feature_cols, MAX_STATE_COLS)
    new_cols = []
    for col in selected:
        values = pd.to_numeric(combined[col], errors='coerce')
        grouped = values.groupby(combined['subject_id'])
        mu = grouped.transform('mean')
        sig = grouped.transform('std').replace(0, np.nan)
        med = grouped.transform('median')
        z_col = f'v47_state_{col}_z'
        abs_col = f'v47_state_{col}_absz'
        delta_col = f'v47_state_{col}_med_delta'
        pct_col = f'v47_state_{col}_pct'
        combined[z_col] = ((values - mu) / sig).clip(-6, 6)
        combined[abs_col] = combined[z_col].abs()
        combined[delta_col] = values - med
        combined[pct_col] = values.groupby(combined['subject_id']).rank(pct=True)
        new_cols.extend([z_col, abs_col, delta_col, pct_col])

    if new_cols:
        abs_cols = [c for c in new_cols if c.endswith('_absz')]
        combined['v47_state_absz_mean'] = combined[abs_cols].mean(axis=1)
        combined['v47_state_absz_max'] = combined[abs_cols].max(axis=1)
        new_cols.extend(['v47_state_absz_mean', 'v47_state_absz_max'])

    train_state = combined.loc[combined['_split'] == 'train', new_cols].reset_index(drop=True)
    test_state = combined.loc[combined['_split'] == 'test', new_cols].reset_index(drop=True)
    print(f'[v47] subject-state columns: {len(new_cols)} from {len(selected)} base columns')
    return (
        pd.concat([train_full.reset_index(drop=True), train_state], axis=1),
        pd.concat([test_full.reset_index(drop=True), test_state], axis=1),
        new_cols,
        selected,
    )


def add_calendar_and_subject_shape(train_full, test_full):
    combined = pd.concat([
        train_full[['subject_id', 'lifelog_date', 'sleep_date']].assign(_split='train'),
        test_full[['subject_id', 'lifelog_date', 'sleep_date']].assign(_split='test'),
    ], ignore_index=True)
    combined = combined.sort_values(['subject_id', 'sleep_date']).reset_index()
    for date_col in ['lifelog_date', 'sleep_date']:
        combined[f'v47_{date_col}_dow'] = combined[date_col].dt.dayofweek
        combined[f'v47_{date_col}_is_weekend'] = (combined[f'v47_{date_col}_dow'] >= 5).astype(int)
        combined[f'v47_{date_col}_dow_sin'] = np.sin(2 * np.pi * combined[f'v47_{date_col}_dow'] / 7)
        combined[f'v47_{date_col}_dow_cos'] = np.cos(2 * np.pi * combined[f'v47_{date_col}_dow'] / 7)

    combined['v47_subject_num'] = combined['subject_id'].str.extract(r'(\d+)').astype(int)
    combined['v47_subject_order'] = combined.groupby('subject_id').cumcount()
    combined['v47_subject_n_rows_all'] = combined.groupby('subject_id')['subject_id'].transform('size')
    combined['v47_subject_pos_frac'] = combined['v47_subject_order'] / (combined['v47_subject_n_rows_all'] - 1).replace(0, np.nan)
    combined['v47_gap_prev_sleep'] = combined.groupby('subject_id')['sleep_date'].diff().dt.days
    combined['v47_gap_next_sleep'] = -combined.groupby('subject_id')['sleep_date'].diff(-1).dt.days
    shape_cols = [c for c in combined.columns if c.startswith('v47_')]
    combined = combined.sort_values('index')
    train_shape = combined.loc[combined['_split'] == 'train', shape_cols].reset_index(drop=True)
    test_shape = combined.loc[combined['_split'] == 'test', shape_cols].reset_index(drop=True)
    return (
        pd.concat([train_full.reset_index(drop=True), train_shape], axis=1),
        pd.concat([test_full.reset_index(drop=True), test_shape], axis=1),
        shape_cols,
    )


def build_feature_table(train, sub):
    if CACHE_FEATURES and FEATURE_CACHE_PATH.exists():
        print(f'[v47] loading cached features: {FEATURE_CACHE_PATH}')
        cached = pd.read_pickle(FEATURE_CACHE_PATH)
        return cached['train_full'], cached['test_full'], cached['feature_cols'], cached['metadata']

    if USE_BASE_V29_FEATURES:
        print('[v47] building v29 base feature table...')
        train_full, test_full, feature_cols = v29.build_feature_table(train, sub)
    else:
        train_full = train.copy()
        test_full = sub[['subject_id', 'lifelog_date', 'sleep_date']].copy()
        feature_cols = []

    extra_train, extra_test = build_hourgrid_feature_table(train, sub)
    train_full = pd.concat([train_full.reset_index(drop=True), extra_train.reset_index(drop=True)], axis=1)
    test_full = pd.concat([test_full.reset_index(drop=True), extra_test.reset_index(drop=True)], axis=1)

    extra_cols = extra_train.columns.tolist()
    train_full, test_full, state_cols, state_source_cols = add_subject_state_features(
        train_full,
        test_full,
        feature_cols + extra_cols,
    )
    train_full, test_full, shape_cols = add_calendar_and_subject_shape(train_full, test_full)

    all_feature_cols = [
        col for col in train_full.columns
        if col not in ['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS
        and pd.api.types.is_numeric_dtype(train_full[col])
    ]
    metadata = {
        'use_base_v29_features': USE_BASE_V29_FEATURES,
        'base_feature_count': len(feature_cols),
        'hourgrid_feature_count': len(extra_cols),
        'state_feature_count': len(state_cols),
        'shape_feature_count': len(shape_cols),
        'state_source_cols': state_source_cols,
        'feature_count': len(all_feature_cols),
    }
    print(f'[v47] total numeric feature columns: {len(all_feature_cols)}')

    if CACHE_FEATURES:
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({
            'train_full': train_full,
            'test_full': test_full,
            'feature_cols': all_feature_cols,
            'metadata': metadata,
        }, FEATURE_CACHE_PATH)
        print(f'[v47] cached features: {FEATURE_CACHE_PATH}')
    return train_full, test_full, all_feature_cols, metadata


def prune_features(x_train, feature_cols):
    selected = []
    for col in feature_cols:
        vals = pd.to_numeric(x_train[col], errors='coerce')
        if vals.notna().mean() < 0.08:
            continue
        if vals.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return selected


def select_model_features(x_train, y, feature_cols, target):
    selected = prune_features(x_train, feature_cols)
    if len(selected) <= MAX_MODEL_FEATURES:
        return selected

    x = x_train[selected]
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=350,
        learning_rate=0.035,
        num_leaves=9,
        min_child_samples=18,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x, y)
    imp = pd.Series(model.feature_importances_, index=selected).sort_values(ascending=False)
    top = imp.loc[imp > 0].head(MAX_MODEL_FEATURES).index.tolist()
    if len(top) < 30:
        top = selected[:MAX_MODEL_FEATURES]
    print(f'  [v47] {target}: selected {len(top)} / {len(selected)} usable features')
    return top


def fit_predict_target(train_full, test_full, feature_cols, target):
    y = train_full[target].to_numpy(dtype=int)
    x_all = train_full[feature_cols].copy()
    x_test_all = test_full[feature_cols].copy()
    selected = select_model_features(x_all, y, feature_cols, target)
    x_all = x_all[selected]
    x_test_all = x_test_all[selected]

    class_counts = np.bincount(y, minlength=2)
    n_folds = int(min(5, class_counts.min()))
    if n_folds < 2:
        mean_prob = float(np.clip(y.mean(), 0.02, 0.98))
        return np.full(len(x_all), mean_prob), np.full(len(x_test_all), mean_prob), selected, {}

    seeds = SEED_POOL[:max(1, min(N_SEEDS, len(SEED_POOL)))]
    oof_lgb = np.zeros(len(x_all), dtype=float)
    oof_ext = np.zeros(len(x_all), dtype=float)
    oof_lr = np.zeros(len(x_all), dtype=float)
    test_lgb = np.zeros(len(x_test_all), dtype=float)
    test_ext = np.zeros(len(x_test_all), dtype=float)
    test_lr = np.zeros(len(x_test_all), dtype=float)

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        seed_oof_lgb = np.zeros(len(x_all), dtype=float)
        seed_oof_ext = np.zeros(len(x_all), dtype=float)
        seed_oof_lr = np.zeros(len(x_all), dtype=float)
        seed_test_lgb = np.zeros(len(x_test_all), dtype=float)
        seed_test_ext = np.zeros(len(x_test_all), dtype=float)
        seed_test_lr = np.zeros(len(x_test_all), dtype=float)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(x_all, y), start=1):
            x_tr = x_all.iloc[tr_idx]
            x_val = x_all.iloc[val_idx]
            y_tr = y[tr_idx]
            y_val = y[val_idx]

            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=N_ESTIMATORS,
                learning_rate=0.025,
                num_leaves=9,
                max_depth=4,
                min_child_samples=16,
                subsample=0.85,
                colsample_bytree=0.65,
                reg_alpha=0.6,
                reg_lambda=4.0,
                random_state=seed + fold,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(
                x_tr,
                y_tr,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
            )
            seed_oof_lgb[val_idx] = lgb_model.predict_proba(x_val)[:, 1]
            seed_test_lgb += lgb_model.predict_proba(x_test_all)[:, 1] / n_folds

            ext_model = ExtraTreesClassifier(
                n_estimators=420,
                max_depth=5,
                min_samples_leaf=7,
                max_features=0.55,
                class_weight='balanced',
                random_state=seed + fold,
                n_jobs=-1,
            )
            imp = SimpleImputer(strategy='median')
            x_tr_imp = imp.fit_transform(x_tr)
            x_val_imp = imp.transform(x_val)
            x_test_imp = imp.transform(x_test_all)
            ext_model.fit(x_tr_imp, y_tr)
            seed_oof_ext[val_idx] = ext_model.predict_proba(x_val_imp)[:, 1]
            seed_test_ext += ext_model.predict_proba(x_test_imp)[:, 1] / n_folds

            lr_model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(
                    C=0.20,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1500,
                    class_weight='balanced',
                )),
            ])
            lr_model.fit(x_tr, y_tr)
            seed_oof_lr[val_idx] = lr_model.predict_proba(x_val)[:, 1]
            seed_test_lr += lr_model.predict_proba(x_test_all)[:, 1] / n_folds

        oof_lgb += seed_oof_lgb / len(seeds)
        oof_ext += seed_oof_ext / len(seeds)
        oof_lr += seed_oof_lr / len(seeds)
        test_lgb += seed_test_lgb / len(seeds)
        test_ext += seed_test_ext / len(seeds)
        test_lr += seed_test_lr / len(seeds)

    raw_oof = clip_prob(0.58 * oof_lgb + 0.27 * oof_ext + 0.15 * oof_lr)
    raw_test = clip_prob(0.58 * test_lgb + 0.27 * test_ext + 0.15 * test_lr)
    diagnostics = {
        'selected_feature_count': len(selected),
        'selected_features_top30': selected[:30],
        'lgb_loss': float(log_loss(y, clip_prob(oof_lgb))),
        'extra_loss': float(log_loss(y, clip_prob(oof_ext))),
        'lr_loss': float(log_loss(y, clip_prob(oof_lr))),
        'raw_loss': float(log_loss(y, raw_oof)),
    }
    return raw_oof, raw_test, selected, diagnostics


def load_anchor():
    oof_path = Path(os.environ.get('V47_ANCHOR_OOF', DEFAULT_ANCHOR_OOF))
    sub_path = Path(os.environ.get('V47_ANCHOR_SUB', DEFAULT_ANCHOR_SUB))
    if not oof_path.exists() or not sub_path.exists():
        print('[v47] default anchor missing; falling back to v38/w40')
        oof_path = FALLBACK_ANCHOR_OOF
        sub_path = FALLBACK_ANCHOR_SUB
    print(f'[v47] anchor_oof={oof_path}')
    print(f'[v47] anchor_sub={sub_path}')
    return load_frame(oof_path), load_frame(sub_path), str(oof_path), str(sub_path)


def build_role_masks(train, sub):
    profiles = interior_eval.build_profiles(train, sub)
    simple_mask, fragmented_mask, all_interior_mask, tail_mask = v45.build_disjoint_proxy_role_masks(train, profiles)
    actual_interior_mask = v45.build_actual_interior_mask(train, sub)
    actual_tail_mask = pd.Series(~actual_interior_mask.to_numpy(), index=sub.index)
    return {
        'simple': simple_mask,
        'fragmented': fragmented_mask,
        'all_interior': all_interior_mask,
        'tail': tail_mask,
        'hidden': all_interior_mask | tail_mask,
        'actual_interior': actual_interior_mask,
        'actual_tail': actual_tail_mask,
    }


def evaluate_frame(train, pred, mask=None):
    if mask is None:
        mask = pd.Series(True, index=train.index)
    per_target = {
        target: float(log_loss(
            train.loc[mask, target].to_numpy(),
            clip_prob(pred.loc[mask, target].to_numpy()),
        ))
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def make_keys(df):
    return df[['subject_id', 'sleep_date', 'lifelog_date']].copy()


def save_prediction_frame(name, train, sub, oof_pred, sub_pred):
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof_pred.to_csv(oof_path, index=False)
    sub_pred.to_csv(sub_path, index=False)
    return str(oof_path), str(sub_path)


def blend_frames(anchor_oof, anchor_sub, raw_oof, raw_sub, weights, roles=None):
    oof = anchor_oof.copy()
    sub = anchor_sub.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        if weight <= 0:
            continue
        if roles is None:
            train_mask = pd.Series(True, index=oof.index)
            sub_mask = pd.Series(True, index=sub.index)
        else:
            train_mask, sub_mask = roles
        oof.loc[train_mask, target] = clip_prob(
            (1.0 - weight) * anchor_oof.loc[train_mask, target]
            + weight * raw_oof.loc[train_mask, target]
        )
        sub.loc[sub_mask, target] = clip_prob(
            (1.0 - weight) * anchor_sub.loc[sub_mask, target]
            + weight * raw_sub.loc[sub_mask, target]
        )
    return oof, sub


def candidate_weights_from_oof(train, anchor_oof, raw_oof):
    weights = {}
    diagnostics = {}
    for target in TARGETS:
        y = train[target].to_numpy()
        anchor_loss = float(log_loss(y, clip_prob(anchor_oof[target])))
        raw_loss = float(log_loss(y, clip_prob(raw_oof[target])))
        delta = raw_loss - anchor_loss
        if target == 'S4':
            weight = 0.0
        elif delta < -0.008:
            weight = 0.15
        elif delta < -0.003:
            weight = 0.10
        elif delta < 0.002:
            weight = 0.05
        else:
            weight = 0.0
        weights[target] = weight
        diagnostics[target] = {
            'anchor_loss': anchor_loss,
            'raw_loss': raw_loss,
            'delta_raw_minus_anchor': delta,
            'weight': weight,
        }
    return weights, diagnostics


def write_blend_candidate(name, train, sub, anchor_oof, anchor_sub, raw_oof, raw_sub, weights, roles=None):
    oof, submission = blend_frames(anchor_oof, anchor_sub, raw_oof, raw_sub, weights, roles=roles)
    oof_path, sub_path = save_prediction_frame(name, train, sub, oof, submission)
    all_loss, per_target = evaluate_frame(train, oof)
    dist = v45.describe_vs_anchor(submission, anchor_sub)
    print(f'[v47] saved {name}: oof={all_loss:.6f} weights={weights} sub={sub_path}')
    return {
        'name': name,
        'weights': weights,
        'oof_loss': all_loss,
        'oof_per_target': per_target,
        'distribution_vs_anchor': dist,
        'oof_path': oof_path,
        'submission': sub_path,
    }


def main():
    ensure_dirs()
    log_path = LOG_DIR / f'run_{EXP_TAG}.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG}...')
    print(f'[v47] config N_SEEDS={N_SEEDS} N_ESTIMATORS={N_ESTIMATORS} MAX_MODEL_FEATURES={MAX_MODEL_FEATURES}')
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_PATH)
    train_full, test_full, feature_cols, feature_metadata = build_feature_table(train, sub)
    anchor_oof, anchor_sub, anchor_oof_path, anchor_sub_path = load_anchor()
    role_masks = build_role_masks(train, sub)

    raw_oof = make_keys(train)
    raw_sub = make_keys(sub)
    target_diagnostics = {}
    selected_features = {}
    for target in TARGETS:
        print(f'\n[v47] training raw hour-grid target={target}')
        oof, pred, selected, diag = fit_predict_target(train_full, test_full, feature_cols, target)
        raw_oof[target] = oof
        raw_sub[target] = pred
        target_diagnostics[target] = diag
        selected_features[target] = selected
        print(f'  raw target loss={diag["raw_loss"]:.6f}')

    raw_name = f'{EXP_TAG}_raw'
    raw_oof_path, raw_sub_path = save_prediction_frame(raw_name, train, sub, raw_oof, raw_sub)
    raw_loss, raw_per_target = evaluate_frame(train, raw_oof)
    anchor_loss, anchor_per_target = evaluate_frame(train, anchor_oof)
    print(f'\n[v47] anchor_oof={anchor_loss:.6f} raw_oof={raw_loss:.6f}')

    candidates = [{
        'name': raw_name,
        'oof_loss': raw_loss,
        'oof_per_target': raw_per_target,
        'oof_path': raw_oof_path,
        'submission': raw_sub_path,
        'weights': {target: 1.0 for target in TARGETS},
        'distribution_vs_anchor': v45.describe_vs_anchor(raw_sub, anchor_sub),
    }]

    for weight in [0.03, 0.05, 0.08, 0.10, 0.15]:
        weights = {target: weight for target in TARGETS}
        candidates.append(write_blend_candidate(
            f'{EXP_TAG}_anchor_blend_w{int(weight * 100):02d}',
            train,
            sub,
            anchor_oof,
            anchor_sub,
            raw_oof,
            raw_sub,
            weights,
        ))

    for weight in [0.05, 0.08, 0.10, 0.15]:
        weights = {target: weight for target in TARGETS}
        weights['S4'] = 0.0
        candidates.append(write_blend_candidate(
            f'{EXP_TAG}_anchor_blend_non_s4_w{int(weight * 100):02d}',
            train,
            sub,
            anchor_oof,
            anchor_sub,
            raw_oof,
            raw_sub,
            weights,
        ))

    adaptive_weights, adaptive_diag = candidate_weights_from_oof(train, anchor_oof, raw_oof)
    candidates.append(write_blend_candidate(
        f'{EXP_TAG}_anchor_blend_target_oofadaptive',
        train,
        sub,
        anchor_oof,
        anchor_sub,
        raw_oof,
        raw_sub,
        adaptive_weights,
    ))

    interior_roles = (role_masks['all_interior'], role_masks['actual_interior'])
    for weight in [0.08, 0.12]:
        weights = {target: weight for target in TARGETS}
        weights['S4'] = 0.0
        candidates.append(write_blend_candidate(
            f'{EXP_TAG}_interior_non_s4_w{int(weight * 100):02d}',
            train,
            sub,
            anchor_oof,
            anchor_sub,
            raw_oof,
            raw_sub,
            weights,
            roles=interior_roles,
        ))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary = {
        'exp_tag': EXP_TAG,
        'config': {
            'n_seeds': N_SEEDS,
            'seed_pool_used': SEED_POOL[:max(1, min(N_SEEDS, len(SEED_POOL)))],
            'n_estimators': N_ESTIMATORS,
            'max_model_features': MAX_MODEL_FEATURES,
            'max_state_cols': MAX_STATE_COLS,
            'max_roll_cols': MAX_ROLL_COLS,
            'use_base_v29_features': USE_BASE_V29_FEATURES,
            'cache_features': CACHE_FEATURES,
        },
        'feature_metadata': feature_metadata,
        'anchor': {
            'oof_path': anchor_oof_path,
            'submission_path': anchor_sub_path,
            'oof_loss': anchor_loss,
            'oof_per_target': anchor_per_target,
        },
        'raw': {
            'oof_loss': raw_loss,
            'oof_per_target': raw_per_target,
            'target_diagnostics': target_diagnostics,
            'selected_features_top50': {
                target: selected_features[target][:50]
                for target in TARGETS
            },
        },
        'adaptive_weight_diagnostics': adaptive_diag,
        'role_rows': {
            'proxy_all_interior': int(role_masks['all_interior'].sum()),
            'proxy_tail': int(role_masks['tail'].sum()),
            'actual_interior': int(role_masks['actual_interior'].sum()),
            'actual_tail': int(role_masks['actual_tail'].sum()),
        },
        'candidates': candidates,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n[v47] summary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

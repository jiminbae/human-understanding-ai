# 배지민 v12코드를 제미나이에 주고 생성
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import json 
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna 진행 상황은 요약해서만 보도록 설정

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
# v12 Advanced: Feature Engineering + Heterogeneous Ensemble
USE_TRAIN_SUBJ_NORM = os.environ.get('V12_TRAIN_NORM', '0') == '1'
USE_RANK_BLEND = os.environ.get('V12_RANK_BLEND', '0') == '1'
USE_FOLD_SAFE_TE = os.environ.get('V12_FOLD_SAFE_TE', '1') == '1'
USE_CALIBRATION = os.environ.get('V12_CALIBRATION', '0') == '1'
CALIBRATION_METHOD = os.environ.get('V12_CALIB_METHOD', 'platt').strip().lower()
if CALIBRATION_METHOD not in {'platt', 'isotonic'}:
    raise ValueError("V12_CALIB_METHOD must be one of: platt, isotonic")
PSEUDO_PUBLIC_TAIL_FRAC = float(os.environ.get('V12_PSEUDO_TAIL_FRAC', '0.2'))
if PSEUDO_PUBLIC_TAIL_FRAC <= 0 or PSEUDO_PUBLIC_TAIL_FRAC >= 1:
    raise ValueError("V12_PSEUDO_TAIL_FRAC must be in (0, 1)")

FORCE_CPU = os.environ.get('V12_FORCE_CPU', '0') == '1'
HAS_CUDA = (not FORCE_CPU) and torch.cuda.is_available() and torch.cuda.device_count() > 0

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'ch2025_data_items'
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR = OUTPUTS_DIR / 'submissions'
REPORT_DIR = OUTPUTS_DIR / 'report'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'

_tag_parts = ['advanced_ensemble', 'feat_eng']
if USE_TRAIN_SUBJ_NORM: _tag_parts.append('trainnorm')
if USE_RANK_BLEND: _tag_parts.append('rankblend')
if USE_FOLD_SAFE_TE: _tag_parts.append('foldsafe_te')
if USE_CALIBRATION: _tag_parts.append(f'calib_{CALIBRATION_METHOD}')

EXP_TAG = '_public_v12_' + ('_'.join(_tag_parts))
OUTPUT_PATH = OUTPUT_DIR / f'submission_v12{EXP_TAG}.csv'
REPORT_PATH = REPORT_DIR / f'report_v12{EXP_TAG}.txt'
SUMMARY_PATH = SUMMARY_DIR / f'summary_v12{EXP_TAG}.json'
OOF_PATH = OOF_DIR / f'oof_v12{EXP_TAG}.csv'
TEST_PREDS_PATH = REPORT_DIR / f'test_preds_v12{EXP_TAG}.csv'
RUN_LOG_PATH = LOG_DIR / f'run_v12{EXP_TAG}.log'

# ---------------------------------------------------------------------
# 유틸리티 및 데이터 추출 함수들은 기존과 동일하게 유지합니다.
# (Tee, ensure_dirs, agg_stats, safe_mean, load_parquet 등)
# (extract_activity, extract_pedo, extract_hr, extract_screen 등 생략 없이 기존 코드 그대로 사용)
# ---------------------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def agg_stats(vals, prefix):
    if len(vals) == 0:
        return {
            f'{prefix}_mean': np.nan, f'{prefix}_std': np.nan,
            f'{prefix}_min': np.nan, f'{prefix}_max': np.nan,
            f'{prefix}_median': np.nan, f'{prefix}_q25': np.nan,
            f'{prefix}_q75': np.nan,
        }
    return {
        f'{prefix}_mean': np.nanmean(vals), f'{prefix}_std': np.nanstd(vals),
        f'{prefix}_min': np.nanmin(vals), f'{prefix}_max': np.nanmax(vals),
        f'{prefix}_median': np.nanmedian(vals), f'{prefix}_q25': np.nanpercentile(vals, 25),
        f'{prefix}_q75': np.nanpercentile(vals, 75),
    }

def safe_mean(vals):
    arr = np.array(vals)
    return np.nanmean(arr) if len(arr) > 0 else np.nan

def load_parquet(name):
    df = pd.read_parquet(DATA_DIR / f'ch2025_{name}.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def extract_activity(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        acts = grp['m_activity'].values
        h = grp['timestamp'].dt.hour.values
        for a in [0, 3, 4, 7, 8]:
            row[f'act_{a}_ratio'] = (acts == a).mean()
        row['act_active_ratio'] = ((acts == 7) | (acts == 8) | (acts == 3)).mean()
        row['act_still_ratio'] = (acts == 0).mean()
        row['act_n_records'] = len(acts)
        for seg, mask in [('morn', (h >= 6) & (h < 12)), ('aftn', (h >= 12) & (h < 18)),
                          ('eve', (h >= 18) & (h < 22)), ('night', (h >= 22) | (h < 6))]:
            s_acts = acts[mask]
            row[f'act_{seg}_active'] = ((s_acts == 7) | (s_acts == 8)).mean() if len(s_acts) > 0 else np.nan
            row[f'act_{seg}_still'] = (s_acts == 0).mean() if len(s_acts) > 0 else np.nan
        pre = acts[(h >= 22) & (h < 24)]
        row['act_presleep_active'] = ((pre == 7) | (pre == 8)).mean() if len(pre) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_pedo(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row['pedo_total_steps'] = grp['step'].sum()
        row['pedo_total_distance'] = grp['distance'].sum()
        row['pedo_total_calories'] = grp['burned_calories'].sum()
        row['pedo_max_speed'] = grp['speed'].max()
        row['pedo_mean_speed'] = grp['speed'].mean()
        row['pedo_running_steps'] = grp['running_step'].sum()
        row['pedo_walking_steps'] = grp['walking_step'].sum()
        row['pedo_run_ratio'] = grp['running_step'].sum() / (grp['step'].sum() + 1)
        eve = grp[grp['timestamp'].dt.hour.between(18, 21)]
        row['pedo_evening_steps'] = eve['step'].sum()
        row['pedo_step_freq_mean'] = grp['step_frequency'].mean()
        row['pedo_step_freq_max'] = grp['step_frequency'].max()
        hourly = grp.groupby(grp['timestamp'].dt.hour)['step'].sum()
        row['pedo_active_hours'] = (hourly > 50).sum()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_hr(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}

        def get_hr_array(series):
            vals = []
            for v in series:
                try:
                    arr = np.asarray(v, dtype=float).ravel()
                    vals.extend(arr[arr > 0].tolist())
                except Exception:
                    if isinstance(v, (int, float)) and v > 0:
                        vals.append(float(v))
            return np.array(vals)

        daily_hr = get_hr_array(grp['heart_rate'])
        if len(daily_hr) > 0:
            row['hr_daily_mean']  = np.nanmean(daily_hr)
            row['hr_daily_std']   = np.nanstd(daily_hr)
            row['hr_daily_min']   = np.nanmin(daily_hr)
            row['hr_daily_max']   = np.nanmax(daily_hr)
            row['hr_daily_rmssd'] = float(np.sqrt(np.nanmean(np.diff(daily_hr) ** 2))) if len(daily_hr) > 1 else np.nan
        else:
            row['hr_daily_mean'] = row['hr_daily_std'] = row['hr_daily_min'] = np.nan
            row['hr_daily_max']  = row['hr_daily_rmssd'] = np.nan

        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('morn', (6, 12)), ('aftn', (12, 18)), ('eve', (18, 22)), ('night', (22, 24))]:
            seg_hr = get_hr_array(grp.loc[h.between(lo, hi - 1), 'heart_rate'])
            row[f'hr_{seg}_mean'] = np.nanmean(seg_hr) if len(seg_hr) > 0 else np.nan
            row[f'hr_{seg}_std']  = np.nanstd(seg_hr)  if len(seg_hr) > 0 else np.nan

        feats.append(row)

    df_feats = pd.DataFrame(feats)

    if 'hr_daily_mean' in df_feats.columns:
        subj_mean = df_feats.groupby('subject_id')['hr_daily_mean'].transform('mean')
        df_feats['hr_daily_rel_mean'] = df_feats['hr_daily_mean'] - subj_mean

    return keys.merge(df_feats, on=['subject_id', 'lifelog_date'], how='left')

def extract_screen(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        sc = grp['m_screen_use'].values
        h = grp['timestamp'].dt.hour.values
        row['screen_on_total'] = (sc > 0).sum()
        row['screen_on_ratio'] = (sc > 0).mean()
        row['screen_unlock_cnt'] = ((sc[1:] > sc[:-1])).sum() if len(sc) > 1 else 0
        for seg, mask in [('night', (h >= 22) | (h < 2)), ('eve', (h >= 20) & (h <= 23)), ('presleep', (h >= 22) & (h < 24))]:
            s_sc = sc[mask]
            row[f'screen_{seg}_on'] = (s_sc > 0).sum()
            row[f'screen_{seg}_ratio'] = (s_sc > 0).mean() if len(s_sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_light(df_raw, col, prefix, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        vals = grp[col].dropna().values
        for k, v in agg_stats(vals, f'{prefix}_all').items():
            row[k] = v
        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('eve', (18, 22)), ('morn', (6, 10)), ('night', (22, 24))]:
            sv = grp.loc[h.between(lo, hi - 1), col].dropna().values
            row[f'{prefix}_{seg}_mean'] = safe_mean(sv)
        row[f'{prefix}_dark_ratio'] = (vals < 10).mean() if len(vals) > 0 else np.nan
        row[f'{prefix}_bright_ratio'] = (vals > 1000).mean() if len(vals) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_ac(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        ch = grp['m_charging'].values
        h = grp['timestamp'].dt.hour.values
        row['ac_charging_ratio'] = ch.mean()
        for seg, mask in [('eve', (h >= 21) & (h <= 23)), ('night', (h >= 22) | (h < 4)), ('presleep', (h >= 22) & (h < 24))]:
            sc = ch[mask]
            row[f'ac_{seg}_charging'] = sc.mean() if len(sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_gps(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        speeds, lats, lons = [], [], []
        for v in grp['m_gps']:
            if isinstance(v, list):
                for pt in v:
                    if isinstance(pt, dict):
                        speeds.append(pt.get('speed', 0))
                        lats.append(pt.get('latitude', 0))
                        lons.append(pt.get('longitude', 0))
        speeds = np.array(speeds)
        row['gps_mean_speed'] = np.nanmean(speeds) if len(speeds) > 0 else np.nan
        row['gps_max_speed'] = np.nanmax(speeds) if len(speeds) > 0 else np.nan
        row['gps_moving_ratio'] = (speeds > 0.5).mean() if len(speeds) > 0 else np.nan
        row['gps_lat_std'] = np.nanstd(lats) if len(lats) > 0 else np.nan
        row['gps_lon_std'] = np.nanstd(lons) if len(lons) > 0 else np.nan
        if len(lats) > 1:
            dlat = np.diff(lats)
            dlon = np.diff(lons)
            row['gps_total_disp'] = float(np.sum(np.sqrt(dlat ** 2 + dlon ** 2)))
        else:
            row['gps_total_disp'] = 0.0
        feats.append(row)
    return pd.DataFrame(feats)

def extract_usage(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        total_time, late_time, eve_time, n_apps = 0, 0, 0, 0
        for ts, v in zip(grp['timestamp'], grp['m_usage_stats']):
            if isinstance(v, list):
                for app in v:
                    if isinstance(app, dict):
                        t = app.get('total_time', 0) or 0
                        total_time += t
                        n_apps += 1
                        if ts.hour >= 22 or ts.hour < 2:
                            late_time += t
                        if ts.hour >= 18:
                            eve_time += t
        row['usage_total_time'] = total_time
        row['usage_n_apps'] = n_apps
        row['usage_late_time'] = late_time
        row['usage_late_ratio'] = late_time / (total_time + 1)
        row['usage_eve_time'] = eve_time
        row['usage_eve_ratio'] = eve_time / (total_time + 1)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_wifi(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        all_bssids, rssi_vals = set(), []
        for v in grp['m_wifi']:
            if isinstance(v, list):
                for net in v:
                    if isinstance(net, dict):
                        all_bssids.add(net.get('bssid', ''))
                        rssi_vals.append(net.get('rssi', -100))
        row['wifi_n_unique'] = len(all_bssids)
        row['wifi_mean_rssi'] = np.mean(rssi_vals) if rssi_vals else np.nan
        row['wifi_max_rssi'] = np.max(rssi_vals) if rssi_vals else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_ble(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        addrs = set()
        for v in grp['m_ble']:
            if isinstance(v, list):
                for dev in v:
                    if isinstance(dev, dict):
                        addrs.add(dev.get('address', ''))
        row['ble_n_unique'] = len(addrs)
        row['ble_n_scans'] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_wlight(df_raw, keys): return extract_light(df_raw, 'w_light', 'wlight', keys)

def extract_ambience(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        music_s, speech_s, silence_s = [], [], []
        for v in grp['m_ambience']:
            if isinstance(v, list):
                d_map = {item[0]: item[1] for item in v if isinstance(item, list) and len(item) == 2}
                music_s.append(d_map.get('Music', 0))
                speech_s.append(d_map.get('Speech', 0))
                silence_s.append(d_map.get('Silence', 0))
        row['amb_music_mean'] = np.mean(music_s) if music_s else np.nan
        row['amb_speech_mean'] = np.mean(speech_s) if speech_s else np.nan
        row['amb_silence_mean'] = np.mean(silence_s) if silence_s else np.nan
        row['amb_n_records'] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_hr(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    df_m = df_raw[df_raw['timestamp'].dt.hour < 9].copy()
    feats = []
    for (sid, d), grp in df_m.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        hour_vals = {h: [] for h in range(9)}
        all_v = []
        for ts, v in zip(grp['timestamp'], grp['heart_rate']):
            try: arr = np.asarray(v, dtype=float).ravel(); arr = arr[arr > 0]
            except Exception: arr = np.array([])
            all_v.extend(arr.tolist()); hour_vals[ts.hour].extend(arr.tolist())
        sleep_hrs = np.array(all_v)
        sleep_hrs = sleep_hrs[sleep_hrs > 0] if len(sleep_hrs) > 0 else sleep_hrs
        for k, v in agg_stats(sleep_hrs, 'slp_hr').items(): row[k] = v
        row['slp_hr_deep_ratio'] = (sleep_hrs < 55).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_awake_ratio'] = (sleep_hrs > 75).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_light_ratio'] = ((sleep_hrs >= 55) & (sleep_hrs <= 75)).mean() if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 1:
            diffs = np.diff(sleep_hrs)
            row['slp_hr_rmssd'] = float(np.sqrt(np.nanmean(diffs ** 2)))
        else: row['slp_hr_rmssd'] = np.nan
        row['slp_hr_n_records'] = len(grp)
        row['slp_hr_early_mean'] = safe_mean(sum([hour_vals[h] for h in range(3)], []))
        row['slp_hr_late_mean'] = safe_mean(sum([hour_vals[h] for h in range(6, 9)], []))
        row['slp_hr_mid_mean'] = safe_mean(sum([hour_vals[h] for h in range(3, 6)], []))
        row['slp_hr_range'] = float(np.ptp(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_median'] = float(np.median(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 5:
            rolling = pd.Series(sleep_hrs).rolling(5, min_periods=1).mean().values
            row['slp_hr_spike_count'] = int((np.abs(sleep_hrs - rolling) > 15).sum())
        else: row['slp_hr_spike_count'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_pedo(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        row['slp_pedo_steps'] = morn['step'].sum()
        row['slp_pedo_active'] = (morn['step'] > 5).sum()
        row['slp_pedo_calories'] = morn['burned_calories'].sum()
        row['slp_pedo_n_records'] = len(morn)
        mid = grp[grp['timestamp'].dt.hour.between(2, 4)]
        row['slp_pedo_mid_steps'] = mid['step'].sum()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_activity(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) == 0:
            row.update({'slp_act_still_ratio': np.nan, 'slp_act_active_ratio': np.nan, 'slp_act_n_records': 0})
        else:
            acts = morn['m_activity'].values
            row['slp_act_still_ratio'] = (acts == 0).mean()
            row['slp_act_active_ratio'] = ((acts == 7) | (acts == 8)).mean()
            row['slp_act_n_records'] = len(acts)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_screen(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            sc = morn['m_screen_use'].values
            row['slp_screen_on'] = (sc > 0).sum()
            row['slp_screen_ratio'] = (sc > 0).mean()
        else:
            row['slp_screen_on'] = np.nan; row['slp_screen_ratio'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_light(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            vals = morn['w_light'].dropna().values
            row['slp_wlight_mean'] = safe_mean(vals)
            row['slp_wlight_dark'] = (vals < 5).mean() if len(vals) > 0 else np.nan
            row['slp_wlight_light'] = (vals > 100).mean() if len(vals) > 0 else np.nan
        else:
            row['slp_wlight_mean'] = np.nan; row['slp_wlight_dark'] = np.nan; row['slp_wlight_light'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def rank_norm(a):
    s = pd.Series(a)
    return (s.rank(method='average').values - 1) / max(len(s) - 1, 1)

def _build_subject_history(history_df, target):
    h = history_df[['subject_id', 'lifelog_date', target]].copy()
    h = h.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    hist = {}
    for sid, grp in h.groupby('subject_id'):
        hist[sid] = {'dates': grp['lifelog_date'].to_numpy(), 'labels': grp[target].to_numpy()}
    return hist

def _encode_from_history(history_map, query_df, windows):
    rows = []
    for sid, d in query_df[['subject_id', 'lifelog_date']].itertuples(index=False):
        if sid not in history_map:
            row = {'te_lag1': np.nan}
            for w in windows: row[f'te_enc{w}'] = np.nan
            rows.append(row)
            continue
        dates = history_map[sid]['dates']
        labels = history_map[sid]['labels']
        k = np.searchsorted(dates, d, side='left')
        past = labels[:k]
        row = {'te_lag1': past[-1] if len(past) > 0 else np.nan}
        for w in windows:
            row[f'te_enc{w}'] = np.nanmean(past[-w:]) if len(past) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=query_df.index)

def build_fold_safe_target_encoding(train_hist_df, tr_query_df, val_query_df, test_query_df, target, windows):
    history_map = _build_subject_history(train_hist_df, target)
    tr_te = _encode_from_history(history_map, tr_query_df, windows)
    val_te = _encode_from_history(history_map, val_query_df, windows)
    test_te = _encode_from_history(history_map, test_query_df, windows)
    return tr_te, val_te, test_te

def calibrate_probs(y_true, oof_prob, test_prob):
    oof_prob = np.clip(oof_prob, 1e-7, 1 - 1e-7)
    test_prob = np.clip(test_prob, 1e-7, 1 - 1e-7)
    if CALIBRATION_METHOD == 'isotonic':
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(oof_prob, y_true)
        cal_oof = cal.transform(oof_prob)
        cal_test = cal.transform(test_prob)
    else:
        cal = LogisticRegression(solver='lbfgs', max_iter=1000)
        cal.fit(oof_prob.reshape(-1, 1), y_true)
        cal_oof = cal.predict_proba(oof_prob.reshape(-1, 1))[:, 1]
        cal_test = cal.predict_proba(test_prob.reshape(-1, 1))[:, 1]
    return np.clip(cal_oof, 1e-7, 1 - 1e-7), np.clip(cal_test, 1e-7, 1 - 1e-7)

def build_pseudo_public_mask(df, tail_frac):
    mask = pd.Series(False, index=df.index)
    for _, grp in df.sort_values(['subject_id', 'lifelog_date']).groupby('subject_id'):
        n = len(grp)
        tail_n = max(1, int(np.ceil(n * tail_frac)))
        idx = grp.index[-tail_n:]
        mask.loc[idx] = True
    return mask.values

# ---------------------------------------------------------------------
# 🚀 1. Feature Engineering 강화 (주기성, 교차 변수, EWMA, Diff 추가)
# ---------------------------------------------------------------------
def build_feature_table(train_df, sub_df):
    all_keys = pd.concat([
        train_df[['subject_id', 'lifelog_date']],
        sub_df[['subject_id', 'lifelog_date']],
    ]).drop_duplicates().reset_index(drop=True)

    sleep_keys = pd.concat([
        train_df[['subject_id', 'sleep_date']],
        sub_df[['subject_id', 'sleep_date']],
    ]).drop_duplicates().reset_index(drop=True)

    print('Extracting daytime features...')
    feat_dfs = []
    for name, fn, col, prefix in [
        ('mActivity', extract_activity, None, None), ('wPedo', extract_pedo, None, None),
        ('wHr', extract_hr, None, None), ('mScreenStatus', extract_screen, None, None),
        ('mLight', extract_light, 'm_light', 'mlight'), ('wLight', extract_wlight, None, None),
        ('mACStatus', extract_ac, None, None), ('mGps', extract_gps, None, None),
        ('mUsageStats', extract_usage, None, None), ('mWifi', extract_wifi, None, None),
        ('mBle', extract_ble, None, None), ('mAmbience', extract_ambience, None, None),
    ]:
        print(f'  {name}...')
        df = load_parquet(name)
        feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))
        del df; gc.collect()

    print('Extracting sleep-date features...')
    sleep_feat_dfs = []
    for name, fn in [
        ('wHr', extract_sleep_hr), ('wPedo', extract_sleep_pedo),
        ('mActivity', extract_sleep_activity), ('mScreenStatus', extract_sleep_screen),
        ('wLight', extract_sleep_light),
    ]:
        print(f'  sleep_morning: {name}...')
        df = load_parquet(name)
        sleep_feat_dfs.append(fn(df, sleep_keys))
        del df; gc.collect()

    sleep_feats = sleep_feat_dfs[0]
    for df in sleep_feat_dfs[1:]:
        sleep_feats = sleep_feats.merge(df, on=['subject_id', 'sleep_date'], how='outer')

    feat_all = feat_dfs[0]
    for df in feat_dfs[1:]:
        feat_all = feat_all.merge(df, on=['subject_id', 'lifelog_date'], how='outer')

    feat_all['dow'] = feat_all['lifelog_date'].dt.dayofweek
    feat_all['month'] = feat_all['lifelog_date'].dt.month
    feat_all['week'] = feat_all['lifelog_date'].dt.isocalendar().week.astype(int)
    feat_all['is_weekend'] = (feat_all['dow'] >= 5).astype(int)
    feat_all['subject_num'] = feat_all['subject_id'].str.extract(r'(\d+)').astype(int)
    
    # [NEW] 주기성(Cyclical) 피처 추가
    feat_all['dow_sin'] = np.sin(2 * np.pi * feat_all['dow'] / 7)
    feat_all['dow_cos'] = np.cos(2 * np.pi * feat_all['dow'] / 7)
    feat_all['month_sin'] = np.sin(2 * np.pi * feat_all['month'] / 12)
    feat_all['month_cos'] = np.cos(2 * np.pi * feat_all['month'] / 12)

    # [NEW] 센서 간 교차(Interaction) 피처
    if 'screen_on_total' in feat_all.columns and 'pedo_total_steps' in feat_all.columns:
        feat_all['screen_per_step'] = feat_all['screen_on_total'] / (feat_all['pedo_total_steps'] + 1)
    if 'pedo_total_calories' in feat_all.columns and 'usage_total_time' in feat_all.columns:
        feat_all['cal_per_usage'] = feat_all['pedo_total_calories'] / (feat_all['usage_total_time'] + 1)

    feat_all = feat_all.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)

    # [NEW] EWMA & Diff 트렌드 추가
    roll_cols = [
        'pedo_total_steps', 'pedo_total_calories', 'pedo_total_distance',
        'screen_on_ratio', 'screen_night_on', 'screen_eve_ratio',
        'act_active_ratio', 'act_still_ratio',
        'mlight_all_mean', 'wlight_all_mean',
        'gps_moving_ratio', 'usage_late_ratio', 'usage_eve_ratio',
        'ac_presleep_charging',
    ]
    for col in roll_cols:
        if col not in feat_all.columns:
            continue
        g = feat_all.groupby('subject_id')[col]
        feat_all[f'{col}_lag1'] = g.shift(1)
        feat_all[f'{col}_lag2'] = g.shift(2)
        feat_all[f'{col}_diff1'] = feat_all[col] - feat_all[f'{col}_lag1'] # 어제 대비 변화량
        feat_all[f'{col}_ewma3'] = g.transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()) # 단기 추세
        feat_all[f'{col}_ewma7'] = g.transform(lambda x: x.shift(1).ewm(span=7, adjust=False).mean()) # 중기 추세
        feat_all[f'{col}_roll3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        feat_all[f'{col}_roll7'] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        feat_all[f'{col}_roll14'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())

    train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
    train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    test_full = sub_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
    test_full = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')

    numeric_cols = feat_all.select_dtypes(include=[np.number]).columns.tolist()
    exclude_from_norm = {'subject_num', 'dow', 'month', 'week', 'is_weekend', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'}
    norm_cols = [c for c in numeric_cols if c not in exclude_from_norm and 'lag' not in c and 'roll' not in c and 'ewma' not in c and 'diff' not in c]

    if not USE_TRAIN_SUBJ_NORM:
        for col in norm_cols:
            mu = feat_all.groupby('subject_id')[col].transform('mean')
            sig = feat_all.groupby('subject_id')[col].transform('std').replace(0, np.nan)
            feat_all[f'{col}_subj_z'] = (feat_all[col] - mu) / sig
        train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
        train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
        test_full = sub_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
        test_full = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    else:
        for col in norm_cols:
            tmp = train_full[['subject_id', col]].copy()
            subj_mu = tmp.groupby('subject_id')[col].mean()
            subj_std = tmp.groupby('subject_id')[col].std().replace(0, np.nan)
            global_mu = tmp[col].mean()
            global_std = tmp[col].std()

            train_mu = train_full['subject_id'].map(subj_mu)
            train_sig = train_full['subject_id'].map(subj_std)
            test_mu = test_full['subject_id'].map(subj_mu).fillna(global_mu)
            test_sig = test_full['subject_id'].map(subj_std).fillna(global_std)

            train_full[f'{col}_subj_z'] = (train_full[col] - train_mu) / train_sig
            test_full[f'{col}_subj_z'] = (test_full[col] - test_mu) / test_sig

    feature_cols = [c for c in train_full.columns if c not in ['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS]
    print(f'Total features: {len(feature_cols)}')
    return train_full, test_full, feature_cols


# ---------------------------------------------------------------------
# 🚀 2. 앙상블 고도화 (LGBM + XGBoost + CatBoost 하드보팅/가중평균)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# 🚀 2 & 3. Optuna 파라미터 튜닝 + 앙상블 스태킹 고도화
# ---------------------------------------------------------------------
def train_and_predict(train_full, test_full, feature_cols):
    X_train_base = train_full[feature_cols].copy()
    X_test_base = test_full[feature_cols].copy()

    # [Level 1] 메타 모델 파라미터 (고정)
    meta_params = {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000}

    seeds = [42, 1234, 9999, 7, 314, 2025, 777, 555]
    n_folds = 5
    n_optuna_trials = 50 # 💡 튜닝 횟수 (50회로 상향: 의미 있는 파라미터 탐색 보장)

    oof_preds = np.zeros((len(X_train_base), len(TARGETS)))
    test_preds = np.zeros((len(X_test_base), len(TARGETS)))
    te_windows = [3, 7, 14, 21]

    for ti, target in enumerate(TARGETS):
        y = train_full[target].values
        print(f'\n{"="*40}\n=== Target: {target} | pos_rate: {y.mean():.3f} ===\n{"="*40}')

        # -----------------------------------------------------
        # 🎯 [OPTUNA TUNING PHASE] 각 타겟에 맞는 최적 파라미터 찾기
        # -----------------------------------------------------
        print(f"  [Optuna] Searching for Golden Parameters (Trials: {n_optuna_trials} per model)...")

        # ✅ 누수 방지: outer fold 0의 train 인덱스만 Optuna에 넘김
        # (main 학습 loop의 validation fold가 튜닝에 노출되지 않도록)
        _tune_outer_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        _tune_tr_idx, _ = next(iter(_tune_outer_skf.split(X_train_base, y)))
        X_tune = X_train_base.iloc[_tune_tr_idx].reset_index(drop=True)
        y_tune = y[_tune_tr_idx]

        tune_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 빠른 튜닝을 위해 3-Fold 사용

        # 1. LightGBM 튜닝
        def objective_lgb(trial):
            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
                'n_estimators': 300, 'verbose': -1, 'n_jobs': -1, 'random_state': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
            }
            if HAS_CUDA: params.update({'device': 'gpu'})

            cv_loss = []
            for tr_idx, val_idx in tune_skf.split(X_tune, y_tune):
                model = lgb.LGBMClassifier(**params)
                try: model.fit(X_tune.iloc[tr_idx], y_tune[tr_idx])
                except:
                    params['device'] = 'cpu'
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_tune.iloc[tr_idx], y_tune[tr_idx])
                preds = model.predict_proba(X_tune.iloc[val_idx])[:, 1]
                cv_loss.append(log_loss(y_tune[val_idx], preds))
            return np.mean(cv_loss)

        # 2. XGBoost 튜닝
        def objective_xgb(trial):
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'logloss',
                'n_estimators': 300, 'n_jobs': -1, 'random_state': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9)
            }
            if HAS_CUDA: params.update({'tree_method': 'hist', 'device': 'cuda'})

            cv_loss = []
            for tr_idx, val_idx in tune_skf.split(X_tune, y_tune):
                model = xgb.XGBClassifier(**params)
                model.fit(X_tune.iloc[tr_idx], y_tune[tr_idx], verbose=False)
                preds = model.predict_proba(X_tune.iloc[val_idx])[:, 1]
                cv_loss.append(log_loss(y_tune[val_idx], preds))
            return np.mean(cv_loss)

        # 3. CatBoost 튜닝
        def objective_cat(trial):
            params = {
                'loss_function': 'Logloss', 'iterations': 300, 'verbose': False, 'thread_count': -1, 'random_seed': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
            }
            if HAS_CUDA: params.update({'task_type': 'GPU'})

            cv_loss = []
            for tr_idx, val_idx in tune_skf.split(X_tune, y_tune):
                model = CatBoostClassifier(**params)
                model.fit(X_tune.iloc[tr_idx], y_tune[tr_idx])
                preds = model.predict_proba(X_tune.iloc[val_idx])[:, 1]
                cv_loss.append(log_loss(y_tune[val_idx], preds))
            return np.mean(cv_loss)

        # 튜닝 실행 및 황금 파라미터 획득
        study_lgb = optuna.create_study(direction='minimize'); study_lgb.optimize(objective_lgb, n_trials=n_optuna_trials)
        study_xgb = optuna.create_study(direction='minimize'); study_xgb.optimize(objective_xgb, n_trials=n_optuna_trials)
        study_cat = optuna.create_study(direction='minimize'); study_cat.optimize(objective_cat, n_trials=n_optuna_trials)

        print(f"    -> [Optuna] Best LGBM Loss: {study_lgb.best_value:.4f}")
        print(f"    -> [Optuna] Best XGB Loss:  {study_xgb.best_value:.4f}")
        print(f"    -> [Optuna] Best CAT Loss:  {study_cat.best_value:.4f}")

        # 찾은 최적 파라미터에 실전용 트리 개수(2000)를 덧씌워 최종 파라미터 세팅
        best_lgb_params = {**study_lgb.best_params, 'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'n_estimators': 2000, 'verbose': -1, 'n_jobs': -1}
        best_xgb_params = {**study_xgb.best_params, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 2000, 'n_jobs': -1}
        best_cat_params = {**study_cat.best_params, 'loss_function': 'Logloss', 'iterations': 2000, 'verbose': False, 'thread_count': -1}
        
        if HAS_CUDA:
            best_lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
            best_xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
            best_cat_params.update({'task_type': 'GPU'})
        else:
            best_lgb_params.update({'device': 'cpu'})

        # -----------------------------------------------------
        # ⚔️ [MAIN TRAINING PHASE] 최적 파라미터로 8-Seed 앙상블 진행
        # -----------------------------------------------------
        print("  [Level 0] Training Main Ensemble with Optimized Parameters...")
        target_oof_lgb = np.zeros(len(X_train_base))
        target_oof_xgb = np.zeros(len(X_train_base))
        target_oof_cat = np.zeros(len(X_train_base))
        target_test_lgb = np.zeros(len(X_test_base))
        target_test_xgb = np.zeros(len(X_test_base))
        target_test_cat = np.zeros(len(X_test_base))

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            seed_oof_lgb, seed_oof_xgb, seed_oof_cat = np.zeros(len(X_train_base)), np.zeros(len(X_train_base)), np.zeros(len(X_train_base))
            seed_test_lgb, seed_test_xgb, seed_test_cat = np.zeros(len(X_test_base)), np.zeros(len(X_test_base)), np.zeros(len(X_test_base))

            for tr_idx, val_idx in skf.split(X_train_base, y):
                X_tr, X_val, X_te = X_train_base.iloc[tr_idx].copy(), X_train_base.iloc[val_idx].copy(), X_test_base.copy()
                y_tr, y_val = y[tr_idx], y[val_idx]

                if USE_FOLD_SAFE_TE:
                    hist_df = train_full.iloc[tr_idx][['subject_id', 'lifelog_date', target]].copy()
                    tr_query, val_query = train_full.iloc[tr_idx][['subject_id', 'lifelog_date']].copy(), train_full.iloc[val_idx][['subject_id', 'lifelog_date']].copy()
                    test_query = test_full[['subject_id', 'lifelog_date']].copy()
                    tr_te, val_te, test_te = build_fold_safe_target_encoding(hist_df, tr_query, val_query, test_query, target, te_windows)
                    X_tr, X_val, X_te = pd.concat([X_tr.reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1), pd.concat([X_val.reset_index(drop=True), val_te.reset_index(drop=True)], axis=1), pd.concat([X_te.reset_index(drop=True), test_te.reset_index(drop=True)], axis=1)

                model_lgb = lgb.LGBMClassifier(**{**best_lgb_params, 'random_state': seed})
                model_xgb = xgb.XGBClassifier(**{**best_xgb_params, 'random_state': seed, 'early_stopping_rounds': 100})
                model_cat = CatBoostClassifier(**{**best_cat_params, 'random_seed': seed})

                try: model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
                except: 
                    cpu_params = dict(best_lgb_params); cpu_params['device'] = 'cpu'
                    model_lgb = lgb.LGBMClassifier(**{**cpu_params, 'random_state': seed})
                    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])

                model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)

                seed_oof_lgb[val_idx], seed_oof_xgb[val_idx], seed_oof_cat[val_idx] = model_lgb.predict_proba(X_val)[:, 1], model_xgb.predict_proba(X_val)[:, 1], model_cat.predict_proba(X_val)[:, 1]
                seed_test_lgb += model_lgb.predict_proba(X_te)[:, 1] / n_folds
                seed_test_xgb += model_xgb.predict_proba(X_te)[:, 1] / n_folds
                seed_test_cat += model_cat.predict_proba(X_te)[:, 1] / n_folds

            target_oof_lgb += seed_oof_lgb / len(seeds); target_oof_xgb += seed_oof_xgb / len(seeds); target_oof_cat += seed_oof_cat / len(seeds)
            target_test_lgb += seed_test_lgb / len(seeds); target_test_xgb += seed_test_xgb / len(seeds); target_test_cat += seed_test_cat / len(seeds)

        # -----------------------------------------------------
        # 🔗 [LEVEL 1] 메타 모델 스태킹
        # -----------------------------------------------------
        print("  [Level 1] Training Meta-Model (Stacking) with Optimized Level 0 outputs...")
        X_meta_train = np.column_stack([target_oof_lgb, target_oof_xgb, target_oof_cat])
        X_meta_test = np.column_stack([target_test_lgb, target_test_xgb, target_test_cat])

        meta_oof, meta_test = np.zeros(len(X_train_base)), np.zeros(len(X_test_base))
        meta_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for meta_tr_idx, meta_val_idx in meta_skf.split(X_meta_train, y):
            meta_model = LogisticRegression(**meta_params)
            meta_model.fit(X_meta_train[meta_tr_idx], y[meta_tr_idx])
            meta_oof[meta_val_idx] = meta_model.predict_proba(X_meta_train[meta_val_idx])[:, 1]
            meta_test += meta_model.predict_proba(X_meta_test)[:, 1] / n_folds

        target_oof, target_test = meta_oof, meta_test
        if USE_CALIBRATION: target_oof, target_test = calibrate_probs(y, target_oof, target_test)

        oof_preds[:, ti], test_preds[:, ti] = target_oof, target_test
        print(f'  🎯 [Level 1] Final Stacked OOF [{target}]: {log_loss(y, oof_preds[:, ti]):.4f}')

    return oof_preds, test_preds

def write_report(report_data):
    lines = []
    lines.append('=' * 80)
    lines.append('Baseline v12 Advanced run report')
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('Base: Advanced Feature Engineering + Triple Ensemble (LGB+XGB+CAT)')
    lines.append(f"  Total OOF: {report_data['avg_oof']:.4f}")
    lines.append(f"  Pseudo-public OOF: {report_data['pseudo_public_oof']:.4f}")
    lines.append(f"  Feature count: {report_data['n_features']}")
    text = '\n'.join(lines)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print('\n' + text)

def main():
    ensure_dirs()
    print('Starting Advanced Training Pipeline...')
    train_df = pd.read_csv(TRAIN_PATH)
    sub_df = pd.read_csv(SUB_PATH)
    train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
    sub_df['lifelog_date'] = pd.to_datetime(sub_df['lifelog_date'])
    train_df['sleep_date'] = pd.to_datetime(train_df['sleep_date'])
    sub_df['sleep_date'] = pd.to_datetime(sub_df['sleep_date'])

    train_full, test_full, feature_cols = build_feature_table(train_df, sub_df)
    oof_preds, test_preds = train_and_predict(train_full, test_full, feature_cols)

    per_target = {}
    for i, t in enumerate(TARGETS):
        per_target[t] = log_loss(train_full[t].values, oof_preds[:, i])
    oof_total = float(np.mean(list(per_target.values())))
    
    pseudo_mask = build_pseudo_public_mask(train_full[['subject_id', 'lifelog_date']], PSEUDO_PUBLIC_TAIL_FRAC)
    pseudo_per_target = {}
    for i, t in enumerate(TARGETS):
        pseudo_per_target[t] = log_loss(train_full.loc[pseudo_mask, t].values, oof_preds[pseudo_mask, i])
    pseudo_oof_total = float(np.mean(list(pseudo_per_target.values())))

    print(f'\n{"=" * 55}')
    print(f'v12 Advanced Total OOF: {oof_total:.4f}')
    print(f'v12 Advanced Pseudo-public OOF: {pseudo_oof_total:.4f}')
    print(f'{"=" * 55}')

    submission = sub_df[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    for i, t in enumerate(TARGETS):
        submission[t] = test_preds[:, i].clip(0.02, 0.98)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f'submission saved: {OUTPUT_PATH}')

if __name__ == '__main__':
    main()# 배지민 v12코드를 제미나이에 주고 생성
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import json
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
# v12 Advanced: Feature Engineering + Heterogeneous Ensemble
USE_TRAIN_SUBJ_NORM = os.environ.get('V12_TRAIN_NORM', '0') == '1'
USE_RANK_BLEND = os.environ.get('V12_RANK_BLEND', '0') == '1'
USE_FOLD_SAFE_TE = os.environ.get('V12_FOLD_SAFE_TE', '1') == '1'
USE_CALIBRATION = os.environ.get('V12_CALIBRATION', '0') == '1'
CALIBRATION_METHOD = os.environ.get('V12_CALIB_METHOD', 'platt').strip().lower()
if CALIBRATION_METHOD not in {'platt', 'isotonic'}:
    raise ValueError("V12_CALIB_METHOD must be one of: platt, isotonic")
PSEUDO_PUBLIC_TAIL_FRAC = float(os.environ.get('V12_PSEUDO_TAIL_FRAC', '0.2'))
if PSEUDO_PUBLIC_TAIL_FRAC <= 0 or PSEUDO_PUBLIC_TAIL_FRAC >= 1:
    raise ValueError("V12_PSEUDO_TAIL_FRAC must be in (0, 1)")

FORCE_CPU = os.environ.get('V12_FORCE_CPU', '0') == '1'
HAS_CUDA = (not FORCE_CPU) and torch.cuda.is_available() and torch.cuda.device_count() > 0

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'ch2025_data_items'
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR = OUTPUTS_DIR / 'submissions'
REPORT_DIR = OUTPUTS_DIR / 'report'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'

_tag_parts = ['advanced_ensemble', 'feat_eng']
if USE_TRAIN_SUBJ_NORM: _tag_parts.append('trainnorm')
if USE_RANK_BLEND: _tag_parts.append('rankblend')
if USE_FOLD_SAFE_TE: _tag_parts.append('foldsafe_te')
if USE_CALIBRATION: _tag_parts.append(f'calib_{CALIBRATION_METHOD}')

EXP_TAG = '_public_v12_' + ('_'.join(_tag_parts))
OUTPUT_PATH = OUTPUT_DIR / f'submission_v12{EXP_TAG}.csv'
REPORT_PATH = REPORT_DIR / f'report_v12{EXP_TAG}.txt'
SUMMARY_PATH = SUMMARY_DIR / f'summary_v12{EXP_TAG}.json'
OOF_PATH = OOF_DIR / f'oof_v12{EXP_TAG}.csv'
TEST_PREDS_PATH = REPORT_DIR / f'test_preds_v12{EXP_TAG}.csv'
RUN_LOG_PATH = LOG_DIR / f'run_v12{EXP_TAG}.log'

# ---------------------------------------------------------------------
# 유틸리티 및 데이터 추출 함수들은 기존과 동일하게 유지합니다.
# (Tee, ensure_dirs, agg_stats, safe_mean, load_parquet 등)
# (extract_activity, extract_pedo, extract_hr, extract_screen 등 생략 없이 기존 코드 그대로 사용)
# ---------------------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def agg_stats(vals, prefix):
    if len(vals) == 0:
        return {
            f'{prefix}_mean': np.nan, f'{prefix}_std': np.nan,
            f'{prefix}_min': np.nan, f'{prefix}_max': np.nan,
            f'{prefix}_median': np.nan, f'{prefix}_q25': np.nan,
            f'{prefix}_q75': np.nan,
        }
    return {
        f'{prefix}_mean': np.nanmean(vals), f'{prefix}_std': np.nanstd(vals),
        f'{prefix}_min': np.nanmin(vals), f'{prefix}_max': np.nanmax(vals),
        f'{prefix}_median': np.nanmedian(vals), f'{prefix}_q25': np.nanpercentile(vals, 25),
        f'{prefix}_q75': np.nanpercentile(vals, 75),
    }

def safe_mean(vals):
    arr = np.array(vals)
    return np.nanmean(arr) if len(arr) > 0 else np.nan

def load_parquet(name):
    df = pd.read_parquet(DATA_DIR / f'ch2025_{name}.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def extract_activity(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        acts = grp['m_activity'].values
        h = grp['timestamp'].dt.hour.values
        for a in [0, 3, 4, 7, 8]:
            row[f'act_{a}_ratio'] = (acts == a).mean()
        row['act_active_ratio'] = ((acts == 7) | (acts == 8) | (acts == 3)).mean()
        row['act_still_ratio'] = (acts == 0).mean()
        row['act_n_records'] = len(acts)
        for seg, mask in [('morn', (h >= 6) & (h < 12)), ('aftn', (h >= 12) & (h < 18)),
                          ('eve', (h >= 18) & (h < 22)), ('night', (h >= 22) | (h < 6))]:
            s_acts = acts[mask]
            row[f'act_{seg}_active'] = ((s_acts == 7) | (s_acts == 8)).mean() if len(s_acts) > 0 else np.nan
            row[f'act_{seg}_still'] = (s_acts == 0).mean() if len(s_acts) > 0 else np.nan
        pre = acts[(h >= 22) & (h < 24)]
        row['act_presleep_active'] = ((pre == 7) | (pre == 8)).mean() if len(pre) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_pedo(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row['pedo_total_steps'] = grp['step'].sum()
        row['pedo_total_distance'] = grp['distance'].sum()
        row['pedo_total_calories'] = grp['burned_calories'].sum()
        row['pedo_max_speed'] = grp['speed'].max()
        row['pedo_mean_speed'] = grp['speed'].mean()
        row['pedo_running_steps'] = grp['running_step'].sum()
        row['pedo_walking_steps'] = grp['walking_step'].sum()
        row['pedo_run_ratio'] = grp['running_step'].sum() / (grp['step'].sum() + 1)
        eve = grp[grp['timestamp'].dt.hour.between(18, 21)]
        row['pedo_evening_steps'] = eve['step'].sum()
        row['pedo_step_freq_mean'] = grp['step_frequency'].mean()
        row['pedo_step_freq_max'] = grp['step_frequency'].max()
        hourly = grp.groupby(grp['timestamp'].dt.hour)['step'].sum()
        row['pedo_active_hours'] = (hourly > 50).sum()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_hr(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}

        def get_hr_array(series):
            vals = []
            for v in series:
                try:
                    arr = np.asarray(v, dtype=float).ravel()
                    vals.extend(arr[arr > 0].tolist())
                except Exception:
                    if isinstance(v, (int, float)) and v > 0:
                        vals.append(float(v))
            return np.array(vals)

        daily_hr = get_hr_array(grp['heart_rate'])
        if len(daily_hr) > 0:
            row['hr_daily_mean']  = np.nanmean(daily_hr)
            row['hr_daily_std']   = np.nanstd(daily_hr)
            row['hr_daily_min']   = np.nanmin(daily_hr)
            row['hr_daily_max']   = np.nanmax(daily_hr)
            row['hr_daily_rmssd'] = float(np.sqrt(np.nanmean(np.diff(daily_hr) ** 2))) if len(daily_hr) > 1 else np.nan
        else:
            row['hr_daily_mean'] = row['hr_daily_std'] = row['hr_daily_min'] = np.nan
            row['hr_daily_max']  = row['hr_daily_rmssd'] = np.nan

        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('morn', (6, 12)), ('aftn', (12, 18)), ('eve', (18, 22)), ('night', (22, 24))]:
            seg_hr = get_hr_array(grp.loc[h.between(lo, hi - 1), 'heart_rate'])
            row[f'hr_{seg}_mean'] = np.nanmean(seg_hr) if len(seg_hr) > 0 else np.nan
            row[f'hr_{seg}_std']  = np.nanstd(seg_hr)  if len(seg_hr) > 0 else np.nan

        feats.append(row)

    df_feats = pd.DataFrame(feats)

    if 'hr_daily_mean' in df_feats.columns:
        subj_mean = df_feats.groupby('subject_id')['hr_daily_mean'].transform('mean')
        df_feats['hr_daily_rel_mean'] = df_feats['hr_daily_mean'] - subj_mean

    return keys.merge(df_feats, on=['subject_id', 'lifelog_date'], how='left')

def extract_screen(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        sc = grp['m_screen_use'].values
        h = grp['timestamp'].dt.hour.values
        row['screen_on_total'] = (sc > 0).sum()
        row['screen_on_ratio'] = (sc > 0).mean()
        row['screen_unlock_cnt'] = ((sc[1:] > sc[:-1])).sum() if len(sc) > 1 else 0
        for seg, mask in [('night', (h >= 22) | (h < 2)), ('eve', (h >= 20) & (h <= 23)), ('presleep', (h >= 22) & (h < 24))]:
            s_sc = sc[mask]
            row[f'screen_{seg}_on'] = (s_sc > 0).sum()
            row[f'screen_{seg}_ratio'] = (s_sc > 0).mean() if len(s_sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_light(df_raw, col, prefix, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        vals = grp[col].dropna().values
        for k, v in agg_stats(vals, f'{prefix}_all').items():
            row[k] = v
        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('eve', (18, 22)), ('morn', (6, 10)), ('night', (22, 24))]:
            sv = grp.loc[h.between(lo, hi - 1), col].dropna().values
            row[f'{prefix}_{seg}_mean'] = safe_mean(sv)
        row[f'{prefix}_dark_ratio'] = (vals < 10).mean() if len(vals) > 0 else np.nan
        row[f'{prefix}_bright_ratio'] = (vals > 1000).mean() if len(vals) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_ac(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        ch = grp['m_charging'].values
        h = grp['timestamp'].dt.hour.values
        row['ac_charging_ratio'] = ch.mean()
        for seg, mask in [('eve', (h >= 21) & (h <= 23)), ('night', (h >= 22) | (h < 4)), ('presleep', (h >= 22) & (h < 24))]:
            sc = ch[mask]
            row[f'ac_{seg}_charging'] = sc.mean() if len(sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_gps(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        speeds, lats, lons = [], [], []
        for v in grp['m_gps']:
            if isinstance(v, list):
                for pt in v:
                    if isinstance(pt, dict):
                        speeds.append(pt.get('speed', 0))
                        lats.append(pt.get('latitude', 0))
                        lons.append(pt.get('longitude', 0))
        speeds = np.array(speeds)
        row['gps_mean_speed'] = np.nanmean(speeds) if len(speeds) > 0 else np.nan
        row['gps_max_speed'] = np.nanmax(speeds) if len(speeds) > 0 else np.nan
        row['gps_moving_ratio'] = (speeds > 0.5).mean() if len(speeds) > 0 else np.nan
        row['gps_lat_std'] = np.nanstd(lats) if len(lats) > 0 else np.nan
        row['gps_lon_std'] = np.nanstd(lons) if len(lons) > 0 else np.nan
        if len(lats) > 1:
            dlat = np.diff(lats)
            dlon = np.diff(lons)
            row['gps_total_disp'] = float(np.sum(np.sqrt(dlat ** 2 + dlon ** 2)))
        else:
            row['gps_total_disp'] = 0.0
        feats.append(row)
    return pd.DataFrame(feats)

def extract_usage(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        total_time, late_time, eve_time, n_apps = 0, 0, 0, 0
        for ts, v in zip(grp['timestamp'], grp['m_usage_stats']):
            if isinstance(v, list):
                for app in v:
                    if isinstance(app, dict):
                        t = app.get('total_time', 0) or 0
                        total_time += t
                        n_apps += 1
                        if ts.hour >= 22 or ts.hour < 2:
                            late_time += t
                        if ts.hour >= 18:
                            eve_time += t
        row['usage_total_time'] = total_time
        row['usage_n_apps'] = n_apps
        row['usage_late_time'] = late_time
        row['usage_late_ratio'] = late_time / (total_time + 1)
        row['usage_eve_time'] = eve_time
        row['usage_eve_ratio'] = eve_time / (total_time + 1)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_wifi(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        all_bssids, rssi_vals = set(), []
        for v in grp['m_wifi']:
            if isinstance(v, list):
                for net in v:
                    if isinstance(net, dict):
                        all_bssids.add(net.get('bssid', ''))
                        rssi_vals.append(net.get('rssi', -100))
        row['wifi_n_unique'] = len(all_bssids)
        row['wifi_mean_rssi'] = np.mean(rssi_vals) if rssi_vals else np.nan
        row['wifi_max_rssi'] = np.max(rssi_vals) if rssi_vals else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_ble(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        addrs = set()
        for v in grp['m_ble']:
            if isinstance(v, list):
                for dev in v:
                    if isinstance(dev, dict):
                        addrs.add(dev.get('address', ''))
        row['ble_n_unique'] = len(addrs)
        row['ble_n_scans'] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_wlight(df_raw, keys): return extract_light(df_raw, 'w_light', 'wlight', keys)

def extract_ambience(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        music_s, speech_s, silence_s = [], [], []
        for v in grp['m_ambience']:
            if isinstance(v, list):
                d_map = {item[0]: item[1] for item in v if isinstance(item, list) and len(item) == 2}
                music_s.append(d_map.get('Music', 0))
                speech_s.append(d_map.get('Speech', 0))
                silence_s.append(d_map.get('Silence', 0))
        row['amb_music_mean'] = np.mean(music_s) if music_s else np.nan
        row['amb_speech_mean'] = np.mean(speech_s) if speech_s else np.nan
        row['amb_silence_mean'] = np.mean(silence_s) if silence_s else np.nan
        row['amb_n_records'] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_hr(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    df_m = df_raw[df_raw['timestamp'].dt.hour < 9].copy()
    feats = []
    for (sid, d), grp in df_m.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        hour_vals = {h: [] for h in range(9)}
        all_v = []
        for ts, v in zip(grp['timestamp'], grp['heart_rate']):
            try: arr = np.asarray(v, dtype=float).ravel(); arr = arr[arr > 0]
            except Exception: arr = np.array([])
            all_v.extend(arr.tolist()); hour_vals[ts.hour].extend(arr.tolist())
        sleep_hrs = np.array(all_v)
        sleep_hrs = sleep_hrs[sleep_hrs > 0] if len(sleep_hrs) > 0 else sleep_hrs
        for k, v in agg_stats(sleep_hrs, 'slp_hr').items(): row[k] = v
        row['slp_hr_deep_ratio'] = (sleep_hrs < 55).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_awake_ratio'] = (sleep_hrs > 75).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_light_ratio'] = ((sleep_hrs >= 55) & (sleep_hrs <= 75)).mean() if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 1:
            diffs = np.diff(sleep_hrs)
            row['slp_hr_rmssd'] = float(np.sqrt(np.nanmean(diffs ** 2)))
        else: row['slp_hr_rmssd'] = np.nan
        row['slp_hr_n_records'] = len(grp)
        row['slp_hr_early_mean'] = safe_mean(sum([hour_vals[h] for h in range(3)], []))
        row['slp_hr_late_mean'] = safe_mean(sum([hour_vals[h] for h in range(6, 9)], []))
        row['slp_hr_mid_mean'] = safe_mean(sum([hour_vals[h] for h in range(3, 6)], []))
        row['slp_hr_range'] = float(np.ptp(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_median'] = float(np.median(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 5:
            rolling = pd.Series(sleep_hrs).rolling(5, min_periods=1).mean().values
            row['slp_hr_spike_count'] = int((np.abs(sleep_hrs - rolling) > 15).sum())
        else: row['slp_hr_spike_count'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_pedo(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        row['slp_pedo_steps'] = morn['step'].sum()
        row['slp_pedo_active'] = (morn['step'] > 5).sum()
        row['slp_pedo_calories'] = morn['burned_calories'].sum()
        row['slp_pedo_n_records'] = len(morn)
        mid = grp[grp['timestamp'].dt.hour.between(2, 4)]
        row['slp_pedo_mid_steps'] = mid['step'].sum()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_activity(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) == 0:
            row.update({'slp_act_still_ratio': np.nan, 'slp_act_active_ratio': np.nan, 'slp_act_n_records': 0})
        else:
            acts = morn['m_activity'].values
            row['slp_act_still_ratio'] = (acts == 0).mean()
            row['slp_act_active_ratio'] = ((acts == 7) | (acts == 8)).mean()
            row['slp_act_n_records'] = len(acts)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_screen(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            sc = morn['m_screen_use'].values
            row['slp_screen_on'] = (sc > 0).sum()
            row['slp_screen_ratio'] = (sc > 0).mean()
        else:
            row['slp_screen_on'] = np.nan; row['slp_screen_ratio'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_light(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            vals = morn['w_light'].dropna().values
            row['slp_wlight_mean'] = safe_mean(vals)
            row['slp_wlight_dark'] = (vals < 5).mean() if len(vals) > 0 else np.nan
            row['slp_wlight_light'] = (vals > 100).mean() if len(vals) > 0 else np.nan
        else:
            row['slp_wlight_mean'] = np.nan; row['slp_wlight_dark'] = np.nan; row['slp_wlight_light'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def rank_norm(a):
    s = pd.Series(a)
    return (s.rank(method='average').values - 1) / max(len(s) - 1, 1)

def _build_subject_history(history_df, target):
    h = history_df[['subject_id', 'lifelog_date', target]].copy()
    h = h.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    hist = {}
    for sid, grp in h.groupby('subject_id'):
        hist[sid] = {'dates': grp['lifelog_date'].to_numpy(), 'labels': grp[target].to_numpy()}
    return hist

def _encode_from_history(history_map, query_df, windows):
    rows = []
    for sid, d in query_df[['subject_id', 'lifelog_date']].itertuples(index=False):
        if sid not in history_map:
            row = {'te_lag1': np.nan}
            for w in windows: row[f'te_enc{w}'] = np.nan
            rows.append(row)
            continue
        dates = history_map[sid]['dates']
        labels = history_map[sid]['labels']
        k = np.searchsorted(dates, d, side='left')
        past = labels[:k]
        row = {'te_lag1': past[-1] if len(past) > 0 else np.nan}
        for w in windows:
            row[f'te_enc{w}'] = np.nanmean(past[-w:]) if len(past) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=query_df.index)

def build_fold_safe_target_encoding(train_hist_df, tr_query_df, val_query_df, test_query_df, target, windows):
    history_map = _build_subject_history(train_hist_df, target)
    tr_te = _encode_from_history(history_map, tr_query_df, windows)
    val_te = _encode_from_history(history_map, val_query_df, windows)
    test_te = _encode_from_history(history_map, test_query_df, windows)
    return tr_te, val_te, test_te

def calibrate_probs(y_true, oof_prob, test_prob):
    oof_prob = np.clip(oof_prob, 1e-7, 1 - 1e-7)
    test_prob = np.clip(test_prob, 1e-7, 1 - 1e-7)
    if CALIBRATION_METHOD == 'isotonic':
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(oof_prob, y_true)
        cal_oof = cal.transform(oof_prob)
        cal_test = cal.transform(test_prob)
    else:
        cal = LogisticRegression(solver='lbfgs', max_iter=1000)
        cal.fit(oof_prob.reshape(-1, 1), y_true)
        cal_oof = cal.predict_proba(oof_prob.reshape(-1, 1))[:, 1]
        cal_test = cal.predict_proba(test_prob.reshape(-1, 1))[:, 1]
    return np.clip(cal_oof, 1e-7, 1 - 1e-7), np.clip(cal_test, 1e-7, 1 - 1e-7)

def build_pseudo_public_mask(df, tail_frac):
    mask = pd.Series(False, index=df.index)
    for _, grp in df.sort_values(['subject_id', 'lifelog_date']).groupby('subject_id'):
        n = len(grp)
        tail_n = max(1, int(np.ceil(n * tail_frac)))
        idx = grp.index[-tail_n:]
        mask.loc[idx] = True
    return mask.values

# ---------------------------------------------------------------------
# 🚀 1. Feature Engineering 강화 (주기성, 교차 변수, EWMA, Diff 추가)
# ---------------------------------------------------------------------
def build_feature_table(train_df, sub_df):
    all_keys = pd.concat([
        train_df[['subject_id', 'lifelog_date']],
        sub_df[['subject_id', 'lifelog_date']],
    ]).drop_duplicates().reset_index(drop=True)

    sleep_keys = pd.concat([
        train_df[['subject_id', 'sleep_date']],
        sub_df[['subject_id', 'sleep_date']],
    ]).drop_duplicates().reset_index(drop=True)

    print('Extracting daytime features...')
    feat_dfs = []
    for name, fn, col, prefix in [
        ('mActivity', extract_activity, None, None), ('wPedo', extract_pedo, None, None),
        ('wHr', extract_hr, None, None), ('mScreenStatus', extract_screen, None, None),
        ('mLight', extract_light, 'm_light', 'mlight'), ('wLight', extract_wlight, None, None),
        ('mACStatus', extract_ac, None, None), ('mGps', extract_gps, None, None),
        ('mUsageStats', extract_usage, None, None), ('mWifi', extract_wifi, None, None),
        ('mBle', extract_ble, None, None), ('mAmbience', extract_ambience, None, None),
    ]:
        print(f'  {name}...')
        df = load_parquet(name)
        feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))
        del df; gc.collect()

    print('Extracting sleep-date features...')
    sleep_feat_dfs = []
    for name, fn in [
        ('wHr', extract_sleep_hr), ('wPedo', extract_sleep_pedo),
        ('mActivity', extract_sleep_activity), ('mScreenStatus', extract_sleep_screen),
        ('wLight', extract_sleep_light),
    ]:
        print(f'  sleep_morning: {name}...')
        df = load_parquet(name)
        sleep_feat_dfs.append(fn(df, sleep_keys))
        del df; gc.collect()

    sleep_feats = sleep_feat_dfs[0]
    for df in sleep_feat_dfs[1:]:
        sleep_feats = sleep_feats.merge(df, on=['subject_id', 'sleep_date'], how='outer')

    feat_all = feat_dfs[0]
    for df in feat_dfs[1:]:
        feat_all = feat_all.merge(df, on=['subject_id', 'lifelog_date'], how='outer')

    feat_all['dow'] = feat_all['lifelog_date'].dt.dayofweek
    feat_all['month'] = feat_all['lifelog_date'].dt.month
    feat_all['week'] = feat_all['lifelog_date'].dt.isocalendar().week.astype(int)
    feat_all['is_weekend'] = (feat_all['dow'] >= 5).astype(int)
    feat_all['subject_num'] = feat_all['subject_id'].str.extract(r'(\d+)').astype(int)
    
    # [NEW] 주기성(Cyclical) 피처 추가
    feat_all['dow_sin'] = np.sin(2 * np.pi * feat_all['dow'] / 7)
    feat_all['dow_cos'] = np.cos(2 * np.pi * feat_all['dow'] / 7)
    feat_all['month_sin'] = np.sin(2 * np.pi * feat_all['month'] / 12)
    feat_all['month_cos'] = np.cos(2 * np.pi * feat_all['month'] / 12)

    # [NEW] 센서 간 교차(Interaction) 피처
    if 'screen_on_total' in feat_all.columns and 'pedo_total_steps' in feat_all.columns:
        feat_all['screen_per_step'] = feat_all['screen_on_total'] / (feat_all['pedo_total_steps'] + 1)
    if 'pedo_total_calories' in feat_all.columns and 'usage_total_time' in feat_all.columns:
        feat_all['cal_per_usage'] = feat_all['pedo_total_calories'] / (feat_all['usage_total_time'] + 1)

    feat_all = feat_all.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)

    # [NEW] EWMA & Diff 트렌드 추가
    roll_cols = [
        'pedo_total_steps', 'pedo_total_calories', 'pedo_total_distance',
        'screen_on_ratio', 'screen_night_on', 'screen_eve_ratio',
        'act_active_ratio', 'act_still_ratio',
        'mlight_all_mean', 'wlight_all_mean',
        'gps_moving_ratio', 'usage_late_ratio', 'usage_eve_ratio',
        'ac_presleep_charging',
    ]
    for col in roll_cols:
        if col not in feat_all.columns:
            continue
        g = feat_all.groupby('subject_id')[col]
        feat_all[f'{col}_lag1'] = g.shift(1)
        feat_all[f'{col}_lag2'] = g.shift(2)
        feat_all[f'{col}_diff1'] = feat_all[col] - feat_all[f'{col}_lag1'] # 어제 대비 변화량
        feat_all[f'{col}_ewma3'] = g.transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()) # 단기 추세
        feat_all[f'{col}_ewma7'] = g.transform(lambda x: x.shift(1).ewm(span=7, adjust=False).mean()) # 중기 추세
        feat_all[f'{col}_roll3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        feat_all[f'{col}_roll7'] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        feat_all[f'{col}_roll14'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())

    train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
    train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    test_full = sub_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
    test_full = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')

    numeric_cols = feat_all.select_dtypes(include=[np.number]).columns.tolist()
    exclude_from_norm = {'subject_num', 'dow', 'month', 'week', 'is_weekend', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'}
    norm_cols = [c for c in numeric_cols if c not in exclude_from_norm and 'lag' not in c and 'roll' not in c and 'ewma' not in c and 'diff' not in c]

    if not USE_TRAIN_SUBJ_NORM:
        for col in norm_cols:
            mu = feat_all.groupby('subject_id')[col].transform('mean')
            sig = feat_all.groupby('subject_id')[col].transform('std').replace(0, np.nan)
            feat_all[f'{col}_subj_z'] = (feat_all[col] - mu) / sig
        train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
        train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
        test_full = sub_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
        test_full = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    else:
        for col in norm_cols:
            tmp = train_full[['subject_id', col]].copy()
            subj_mu = tmp.groupby('subject_id')[col].mean()
            subj_std = tmp.groupby('subject_id')[col].std().replace(0, np.nan)
            global_mu = tmp[col].mean()
            global_std = tmp[col].std()

            train_mu = train_full['subject_id'].map(subj_mu)
            train_sig = train_full['subject_id'].map(subj_std)
            test_mu = test_full['subject_id'].map(subj_mu).fillna(global_mu)
            test_sig = test_full['subject_id'].map(subj_std).fillna(global_std)

            train_full[f'{col}_subj_z'] = (train_full[col] - train_mu) / train_sig
            test_full[f'{col}_subj_z'] = (test_full[col] - test_mu) / test_sig

    feature_cols = [c for c in train_full.columns if c not in ['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS]
    print(f'Total features: {len(feature_cols)}')
    return train_full, test_full, feature_cols


# ---------------------------------------------------------------------
# 🚀 2. 앙상블 고도화 (LGBM + XGBoost + CatBoost 하드보팅/가중평균)
# ---------------------------------------------------------------------
def train_and_predict(train_full, test_full, feature_cols):
    X_train_base = train_full[feature_cols].copy()
    X_test_base = test_full[feature_cols].copy()

    # -----------------------------------------------------
    # [Level 0] 기본 부스팅 모델 파라미터 세팅
    # -----------------------------------------------------
    lgb_params_base = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.02, 'feature_fraction': 0.7,
        'bagging_fraction': 0.7, 'bagging_freq': 5, 'min_child_samples': 20,
        'reg_alpha': 0.3, 'reg_lambda': 2.0, 'n_estimators': 2000,
        'verbose': -1, 'n_jobs': -1,
    }
    
    xgb_params_base = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'learning_rate': 0.02, 'max_depth': 6, 'subsample': 0.7,
        'colsample_bytree': 0.7, 'n_estimators': 2000, 'n_jobs': -1,
        'early_stopping_rounds': 100 
    }
    
    cat_params_base = {
        'loss_function': 'Logloss', 'learning_rate': 0.02, 'depth': 6,
        'iterations': 2000, 'verbose': False, 'thread_count': -1,
    }

    if HAS_CUDA:
        lgb_params_base.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
        xgb_params_base.update({'tree_method': 'hist', 'device': 'cuda'})
        cat_params_base.update({'task_type': 'GPU'})
    else:
        lgb_params_base.update({'device': 'cpu'})

    # -----------------------------------------------------
    # [Level 1] 메타 모델 파라미터 세팅 (과적합 방지를 위해 로지스틱 회귀 사용)
    # -----------------------------------------------------
    meta_params = {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000
    }

    seeds = [42, 1234, 9999, 7, 314, 2025, 777, 555]
    n_folds = 5

    oof_preds = np.zeros((len(X_train_base), len(TARGETS)))
    test_preds = np.zeros((len(X_test_base), len(TARGETS)))
    te_windows = [3, 7, 14, 21]

    for ti, target in enumerate(TARGETS):
        y = train_full[target].values
        print(f'\n=== Target: {target} | pos_rate: {y.mean():.3f} ===')

        # 3개 모델의 앙상블 전, 순수 예측 확률을 담을 그릇 (Level 1의 입력값이 됨)
        target_oof_lgb = np.zeros(len(X_train_base))
        target_oof_xgb = np.zeros(len(X_train_base))
        target_oof_cat = np.zeros(len(X_train_base))

        target_test_lgb = np.zeros(len(X_test_base))
        target_test_xgb = np.zeros(len(X_test_base))
        target_test_cat = np.zeros(len(X_test_base))

        # --- LEVEL 0: 개별 부스팅 모델 학습 ---
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            seed_oof_lgb = np.zeros(len(X_train_base))
            seed_oof_xgb = np.zeros(len(X_train_base))
            seed_oof_cat = np.zeros(len(X_train_base))

            seed_test_lgb = np.zeros(len(X_test_base))
            seed_test_xgb = np.zeros(len(X_test_base))
            seed_test_cat = np.zeros(len(X_test_base))

            for tr_idx, val_idx in skf.split(X_train_base, y):
                X_tr = X_train_base.iloc[tr_idx].copy()
                X_val = X_train_base.iloc[val_idx].copy()
                X_te = X_test_base.copy()
                y_tr, y_val = y[tr_idx], y[val_idx]

                if USE_FOLD_SAFE_TE:
                    hist_df = train_full.iloc[tr_idx][['subject_id', 'lifelog_date', target]].copy()
                    tr_query = train_full.iloc[tr_idx][['subject_id', 'lifelog_date']].copy()
                    val_query = train_full.iloc[val_idx][['subject_id', 'lifelog_date']].copy()
                    test_query = test_full[['subject_id', 'lifelog_date']].copy()

                    tr_te, val_te, test_te = build_fold_safe_target_encoding(
                        hist_df, tr_query, val_query, test_query, target, te_windows)

                    X_tr = pd.concat([X_tr.reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
                    X_val = pd.concat([X_val.reset_index(drop=True), val_te.reset_index(drop=True)], axis=1)
                    X_te = pd.concat([X_te.reset_index(drop=True), test_te.reset_index(drop=True)], axis=1)

                # 개별 모델 선언 및 학습
                model_lgb = lgb.LGBMClassifier(**{**lgb_params_base, 'random_state': seed})
                model_xgb = xgb.XGBClassifier(**{**xgb_params_base, 'random_state': seed})
                model_cat = CatBoostClassifier(**{**cat_params_base, 'random_seed': seed})

                try:
                    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
                except Exception:
                    cpu_params = dict(lgb_params_base); cpu_params['device'] = 'cpu'
                    model_lgb = lgb.LGBMClassifier(**{**cpu_params, 'random_state': seed})
                    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])

                model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)

                # 각 모델별 확률 따로 저장 (합치지 않음!)
                seed_oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
                seed_oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
                seed_oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
                
                seed_test_lgb += model_lgb.predict_proba(X_te)[:, 1] / n_folds
                seed_test_xgb += model_xgb.predict_proba(X_te)[:, 1] / n_folds
                seed_test_cat += model_cat.predict_proba(X_te)[:, 1] / n_folds

            # 시드별 결과 평균내어 Level 0 그릇에 차곡차곡 담기
            target_oof_lgb += seed_oof_lgb / len(seeds)
            target_oof_xgb += seed_oof_xgb / len(seeds)
            target_oof_cat += seed_oof_cat / len(seeds)

            target_test_lgb += seed_test_lgb / len(seeds)
            target_test_xgb += seed_test_xgb / len(seeds)
            target_test_cat += seed_test_cat / len(seeds)

        # 각 단일 모델의 OOF 성능 출력
        print(f"  [Level 0] LogLoss - LGBM: {log_loss(y, target_oof_lgb):.4f}, XGB: {log_loss(y, target_oof_xgb):.4f}, CAT: {log_loss(y, target_oof_cat):.4f}")

        # --- LEVEL 1: 메타 모델 스태킹 ---
        print("  [Level 1] Training Meta-Model (Stacking)...")
        # 3개 모델의 확률을 가로로 이어 붙여 새로운 특징(Feature) 3개짜리 데이터셋 생성
        X_meta_train = np.column_stack([target_oof_lgb, target_oof_xgb, target_oof_cat])
        X_meta_test = np.column_stack([target_test_lgb, target_test_xgb, target_test_cat])

        meta_oof = np.zeros(len(X_train_base))
        meta_test = np.zeros(len(X_test_base))

        # 메타 모델 또한 과적합을 막기 위해 5-Fold로 훈련 및 예측
        meta_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for meta_tr_idx, meta_val_idx in meta_skf.split(X_meta_train, y):
            X_meta_tr, X_meta_val = X_meta_train[meta_tr_idx], X_meta_train[meta_val_idx]
            y_meta_tr, y_meta_val = y[meta_tr_idx], y[meta_val_idx]

            meta_model = LogisticRegression(**meta_params)
            meta_model.fit(X_meta_tr, y_meta_tr)

            meta_oof[meta_val_idx] = meta_model.predict_proba(X_meta_val)[:, 1]
            meta_test += meta_model.predict_proba(X_meta_test)[:, 1] / n_folds

        target_oof = meta_oof
        target_test = meta_test

        if USE_CALIBRATION:
            target_oof, target_test = calibrate_probs(y, target_oof, target_test)

        oof_preds[:, ti] = target_oof
        test_preds[:, ti] = target_test

        print(f'  [Level 1] Final Stacked OOF [{target}]: {log_loss(y, oof_preds[:, ti]):.4f}')

    return oof_preds, test_preds

def write_report(report_data):
    lines = []
    lines.append('=' * 80)
    lines.append('Baseline v12 Advanced run report')
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('Base: Advanced Feature Engineering + Triple Ensemble (LGB+XGB+CAT)')
    lines.append(f"  Total OOF: {report_data['avg_oof']:.4f}")
    lines.append(f"  Pseudo-public OOF: {report_data['pseudo_public_oof']:.4f}")
    lines.append(f"  Feature count: {report_data['n_features']}")
    text = '\n'.join(lines)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print('\n' + text)
# ---------------------------------------------------------------------
# 🚀 3. 고급 피처 셀렉션 (노이즈 컷다운)
# ---------------------------------------------------------------------
def perform_feature_selection(train_full, feature_cols, targets, drop_ratio=0.15):
    print("\n[Feature Selection] Evaluating feature importance to cut noise...")
    
    # 컷다운을 위한 가벼운 정찰대(LightGBM) 세팅
    lgb_params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting_type': 'gbdt', 'learning_rate': 0.05,
        'n_estimators': 150, 'verbose': -1, 'n_jobs': -1,
        'random_state': 42
    }
    
    if HAS_CUDA:
        lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
    else:
        lgb_params.update({'device': 'cpu'})
        
    importance_df = pd.DataFrame({'feature': feature_cols})
    importance_df['importance'] = 0.0
    
    # 7개의 타겟(Q1~S4)에 대해 각각 빠르게 학습해보고 진짜 쓸모있는 피처인지 평가
    for t in targets:
        y = train_full[t].values
        model = lgb.LGBMClassifier(**lgb_params)
        
        try:
            model.fit(train_full[feature_cols], y)
        except Exception:
            cpu_params = dict(lgb_params)
            cpu_params['device'] = 'cpu'
            model = lgb.LGBMClassifier(**cpu_params)
            model.fit(train_full[feature_cols], y)
            
        # 트리를 나눌 때(Split) 기여한 정보 획득량(Gain) 누적
        importance_df['importance'] += model.feature_importances_ / len(targets)
        
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # 하위 X%의 잉여 피처와, 아예 기여도가 0인 피처를 색출
    n_drop = int(len(feature_cols) * drop_ratio)
    drop_features = importance_df.tail(n_drop)['feature'].tolist()
    zero_imp_features = importance_df[importance_df['importance'] == 0]['feature'].tolist()
    
    # 중복 제거 후 최종 퇴출 명단 작성
    final_drop_list = list(set(drop_features + zero_imp_features))
    selected_features = [c for c in feature_cols if c not in final_drop_list]
    
    print(f"  - Original features: {len(feature_cols)}개")
    print(f"  - Dropped noise features: {len(final_drop_list)}개 (하위 {drop_ratio*100}% 및 기여도 0)")
    print(f"  - Selected elite features: {len(selected_features)}개")
    
    return selected_features

def main():
    ensure_dirs()
    print('Starting Advanced Training Pipeline...')
    
    # --- 데이터 불러오기 ---
    train_df = pd.read_csv(TRAIN_PATH)
    sub_df = pd.read_csv(SUB_PATH)
    train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
    sub_df['lifelog_date'] = pd.to_datetime(sub_df['lifelog_date'])
    train_df['sleep_date'] = pd.to_datetime(train_df['sleep_date'])
    sub_df['sleep_date'] = pd.to_datetime(sub_df['sleep_date'])

    # 1단계: 모든 피처 생성
    train_full, test_full, feature_cols = build_feature_table(train_df, sub_df)
    
    # [NEW] 2단계: 본격적인 앙상블 학습 전, 노이즈 피처 컷다운 실행
    # (하위 15% 및 기여도 0인 노이즈 피처 제거)
    elite_feature_cols = perform_feature_selection(train_full, feature_cols, TARGETS, drop_ratio=0.15)
    
    # 3단계: 걸러진 정예 피처(elite_feature_cols)만 데리고 메인 학습(스태킹) 시작!
    oof_preds, test_preds = train_and_predict(train_full, test_full, elite_feature_cols)

    # --- 점수 계산 및 검증 ---
    per_target = {}
    for i, t in enumerate(TARGETS):
        per_target[t] = log_loss(train_full[t].values, oof_preds[:, i])
    oof_total = float(np.mean(list(per_target.values())))
    
    pseudo_mask = build_pseudo_public_mask(train_full[['subject_id', 'lifelog_date']], PSEUDO_PUBLIC_TAIL_FRAC)
    pseudo_per_target = {}
    for i, t in enumerate(TARGETS):
        pseudo_per_target[t] = log_loss(train_full.loc[pseudo_mask, t].values, oof_preds[pseudo_mask, i])
    pseudo_oof_total = float(np.mean(list(pseudo_per_target.values())))

    print(f'\n{"=" * 55}')
    print(f'v12 Advanced Total OOF: {oof_total:.4f}')
    print(f'v12 Advanced Pseudo-public OOF: {pseudo_oof_total:.4f}')
    print(f'{"=" * 55}')

    # --- 결과물 저장 (제출 파일 생성) ---
    submission = sub_df[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    for i, t in enumerate(TARGETS):
        submission[t] = test_preds[:, i].clip(0.02, 0.98)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f'submission saved: {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
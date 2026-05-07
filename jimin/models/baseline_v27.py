# v27: v23 + Forward Features + Pseudo-Labeling
#   - add_forward_features(): 같은 subject의 미래 훈련 데이터로 forward rolling 피처 생성
#     (test block 이후 train block이 존재하는 interleaved 구조 활용)
#   - augment_with_pseudo_labels(): v23 test 예측 중 confidence 높은 것을 hard label로
#     원본 450행 + pseudo 행 합쳐 학습; OOF는 원본 행만으로 계산
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
USE_TRAIN_SUBJ_NORM    = os.environ.get('V12_TRAIN_NORM',      '0') == '1'
USE_FOLD_SAFE_TE       = os.environ.get('V12_FOLD_SAFE_TE',    '1') == '1'
USE_CALIBRATION        = os.environ.get('V12_CALIBRATION',     '0') == '1'
CALIBRATION_METHOD     = os.environ.get('V12_CALIB_METHOD', 'platt').strip().lower()
if CALIBRATION_METHOD not in {'platt', 'isotonic'}:
    raise ValueError("V12_CALIB_METHOD must be one of: platt, isotonic")
PSEUDO_PUBLIC_TAIL_FRAC = float(os.environ.get('V12_PSEUDO_TAIL_FRAC', '0.2'))
if not (0 < PSEUDO_PUBLIC_TAIL_FRAC < 1):
    raise ValueError("V12_PSEUDO_TAIL_FRAC must be in (0, 1)")

FORCE_CPU = os.environ.get('V12_FORCE_CPU', '0') == '1'
HAS_CUDA  = (not FORCE_CPU) and torch.cuda.is_available() and torch.cuda.device_count() > 0

STABILITY_THRESHOLD = float(os.environ.get('V23_STAB_THRESHOLD', '0.6'))
MAX_STABLE_FEATURES = int(os.environ.get('V23_MAX_FEATURES', '150'))

PSEUDO_CONFIDENCE = float(os.environ.get('V27_PSEUDO_CONF', '0.7'))
PSEUDO_SOURCE     = os.environ.get('V27_PSEUDO_SOURCE', 'submission_v23_stability_filter.csv')
FWD_WINDOWS       = [7, 14]

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / 'ch2025_data_items'
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH   = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR  = OUTPUTS_DIR / 'submissions'
REPORT_DIR  = OUTPUTS_DIR / 'report'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
OOF_DIR     = OUTPUTS_DIR / 'oof'
LOG_DIR     = OUTPUTS_DIR / 'log'

EXP_TAG      = 'v27_pseudo_fwd'
OUTPUT_PATH  = OUTPUT_DIR  / f'submission_{EXP_TAG}.csv'
RUN_LOG_PATH = LOG_DIR     / f'run_{EXP_TAG}.log'


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass

def ensure_dirs():
    for d in [OUTPUT_DIR, REPORT_DIR, SUMMARY_DIR, OOF_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def agg_stats(vals, prefix):
    if len(vals) == 0:
        return {f'{prefix}_{k}': np.nan
                for k in ['mean','std','min','max','median','q25','q75']}
    return {
        f'{prefix}_mean':   np.nanmean(vals),
        f'{prefix}_std':    np.nanstd(vals),
        f'{prefix}_min':    np.nanmin(vals),
        f'{prefix}_max':    np.nanmax(vals),
        f'{prefix}_median': np.nanmedian(vals),
        f'{prefix}_q25':    np.nanpercentile(vals, 25),
        f'{prefix}_q75':    np.nanpercentile(vals, 75),
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
        h    = grp['timestamp'].dt.hour.values
        for a in [0, 3, 4, 7, 8]: row[f'act_{a}_ratio'] = (acts == a).mean()
        row['act_active_ratio'] = ((acts==7)|(acts==8)|(acts==3)).mean()
        row['act_still_ratio']  = (acts == 0).mean()
        row['act_n_records']    = len(acts)
        for seg, mask in [('morn',(h>=6)&(h<12)),('aftn',(h>=12)&(h<18)),
                          ('eve',(h>=18)&(h<22)),('night',(h>=22)|(h<6))]:
            s = acts[mask]
            row[f'act_{seg}_active'] = ((s==7)|(s==8)).mean() if len(s)>0 else np.nan
            row[f'act_{seg}_still']  = (s==0).mean()          if len(s)>0 else np.nan
        pre = acts[(h>=22)&(h<24)]
        row['act_presleep_active'] = ((pre==7)|(pre==8)).mean() if len(pre)>0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_pedo(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row['pedo_total_steps']    = grp['step'].sum()
        row['pedo_total_distance'] = grp['distance'].sum()
        row['pedo_total_calories'] = grp['burned_calories'].sum()
        row['pedo_max_speed']      = grp['speed'].max()
        row['pedo_mean_speed']     = grp['speed'].mean()
        row['pedo_running_steps']  = grp['running_step'].sum()
        row['pedo_walking_steps']  = grp['walking_step'].sum()
        row['pedo_run_ratio']      = grp['running_step'].sum() / (grp['step'].sum() + 1)
        eve = grp[grp['timestamp'].dt.hour.between(18, 21)]
        row['pedo_evening_steps']  = eve['step'].sum()
        row['pedo_step_freq_mean'] = grp['step_frequency'].mean()
        row['pedo_step_freq_max']  = grp['step_frequency'].max()
        hourly = grp.groupby(grp['timestamp'].dt.hour)['step'].sum()
        row['pedo_active_hours']   = (hourly > 50).sum()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_hr(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    df_day = df_raw[df_raw['timestamp'].dt.hour >= 6].copy()
    feats = []
    for (sid, d), grp in df_day.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        all_hr = []
        seg_hr = {'morn': [], 'aftn': [], 'eve': [], 'presleep': []}
        for ts, v in zip(grp['timestamp'], grp['heart_rate']):
            try:
                arr = np.asarray(v, dtype=float).ravel()
                arr = arr[arr > 0]
            except Exception:
                arr = np.array([])
            all_hr.extend(arr.tolist())
            h = ts.hour
            if   6 <= h < 12: seg_hr['morn'].extend(arr.tolist())
            elif 12 <= h < 18: seg_hr['aftn'].extend(arr.tolist())
            elif 18 <= h < 22: seg_hr['eve'].extend(arr.tolist())
            elif h >= 22:      seg_hr['presleep'].extend(arr.tolist())

        hr = np.array(all_hr)
        hr = hr[hr > 0] if len(hr) > 0 else hr

        for k, v in agg_stats(hr, 'hr_day').items(): row[k] = v
        row['hr_day_resting_ratio']  = (hr < 60).mean()               if len(hr) > 0 else np.nan
        row['hr_day_light_ratio']    = ((hr>=60)&(hr<100)).mean()      if len(hr) > 0 else np.nan
        row['hr_day_moderate_ratio'] = ((hr>=100)&(hr<140)).mean()     if len(hr) > 0 else np.nan
        row['hr_day_vigorous_ratio'] = (hr >= 140).mean()              if len(hr) > 0 else np.nan
        row['hr_day_n_records']      = len(grp)
        row['hr_day_rmssd']          = float(np.sqrt(np.nanmean(np.diff(hr)**2))) if len(hr) > 5 else np.nan

        morn_arr = np.array(seg_hr['morn']); morn_arr = morn_arr[morn_arr > 0] if len(morn_arr) > 0 else morn_arr
        aftn_arr = np.array(seg_hr['aftn']); aftn_arr = aftn_arr[aftn_arr > 0] if len(aftn_arr) > 0 else aftn_arr
        eve_arr  = np.array(seg_hr['eve']);  eve_arr  = eve_arr[eve_arr > 0]   if len(eve_arr) > 0 else eve_arr
        pre_arr  = np.array(seg_hr['presleep']); pre_arr = pre_arr[pre_arr > 0] if len(pre_arr) > 0 else pre_arr

        for seg, arr in [('morn',morn_arr),('aftn',aftn_arr),('eve',eve_arr),('presleep',pre_arr)]:
            row[f'hr_{seg}_mean'] = np.nanmean(arr) if len(arr) > 0 else np.nan
            row[f'hr_{seg}_std']  = np.nanstd(arr)  if len(arr) > 0 else np.nan

        morn_mean = np.nanmean(morn_arr) if len(morn_arr) > 0 else np.nan
        eve_mean  = np.nanmean(eve_arr)  if len(eve_arr)  > 0 else np.nan
        pre_mean  = np.nanmean(pre_arr)  if len(pre_arr)  > 0 else np.nan
        row['hr_morn_eve_diff']     = morn_mean - eve_mean if not (np.isnan(morn_mean) or np.isnan(eve_mean)) else np.nan
        row['hr_eve_presleep_diff'] = eve_mean - pre_mean  if not (np.isnan(eve_mean)  or np.isnan(pre_mean))  else np.nan

        feats.append(row)

    if not feats:
        return pd.DataFrame({'subject_id': keys['subject_id'], 'lifelog_date': keys['lifelog_date']})
    return pd.DataFrame(feats)

def extract_screen(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        sc = grp['m_screen_use'].values
        h  = grp['timestamp'].dt.hour.values
        row['screen_on_total']   = (sc > 0).sum()
        row['screen_on_ratio']   = (sc > 0).mean()
        row['screen_unlock_cnt'] = ((sc[1:] > sc[:-1])).sum() if len(sc) > 1 else 0
        for seg, mask in [('night',(h>=22)|(h<2)),('eve',(h>=20)&(h<=23)),('presleep',(h>=22)&(h<24))]:
            s = sc[mask]
            row[f'screen_{seg}_on']    = (s > 0).sum()
            row[f'screen_{seg}_ratio'] = (s > 0).mean() if len(s) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_light(df_raw, col, prefix, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        vals = grp[col].dropna().values
        for k, v in agg_stats(vals, f'{prefix}_all').items(): row[k] = v
        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('eve',(18,22)),('morn',(6,10)),('night',(22,24))]:
            row[f'{prefix}_{seg}_mean'] = safe_mean(grp.loc[h.between(lo, hi-1), col].dropna().values)
        row[f'{prefix}_dark_ratio']   = (vals < 10).mean()   if len(vals) > 0 else np.nan
        row[f'{prefix}_bright_ratio'] = (vals > 1000).mean() if len(vals) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_ac(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        ch = grp['m_charging'].values
        h  = grp['timestamp'].dt.hour.values
        row['ac_charging_ratio'] = ch.mean()
        for seg, mask in [('eve',(h>=21)&(h<=23)),('night',(h>=22)|(h<4)),('presleep',(h>=22)&(h<24))]:
            s = ch[mask]
            row[f'ac_{seg}_charging'] = s.mean() if len(s) > 0 else np.nan
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
        row['gps_mean_speed']   = np.nanmean(speeds) if len(speeds) > 0 else np.nan
        row['gps_max_speed']    = np.nanmax(speeds)  if len(speeds) > 0 else np.nan
        row['gps_moving_ratio'] = (speeds > 0.5).mean() if len(speeds) > 0 else np.nan
        row['gps_lat_std']      = np.nanstd(lats)    if len(lats) > 0 else np.nan
        row['gps_lon_std']      = np.nanstd(lons)    if len(lons) > 0 else np.nan
        row['gps_total_disp']   = float(np.sum(np.sqrt(np.diff(lats)**2 + np.diff(lons)**2))) if len(lats) > 1 else 0.0
        feats.append(row)
    return pd.DataFrame(feats)

def extract_usage(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        total_time = late_time = eve_time = n_apps = 0
        for ts, v in zip(grp['timestamp'], grp['m_usage_stats']):
            if isinstance(v, list):
                for app in v:
                    if isinstance(app, dict):
                        t = app.get('total_time', 0) or 0
                        total_time += t; n_apps += 1
                        if ts.hour >= 22 or ts.hour < 2: late_time += t
                        if ts.hour >= 18: eve_time += t
        row['usage_total_time'] = total_time
        row['usage_n_apps']     = n_apps
        row['usage_late_time']  = late_time
        row['usage_late_ratio'] = late_time / (total_time + 1)
        row['usage_eve_time']   = eve_time
        row['usage_eve_ratio']  = eve_time / (total_time + 1)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_wifi(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        bssids, rssi = set(), []
        for v in grp['m_wifi']:
            if isinstance(v, list):
                for net in v:
                    if isinstance(net, dict):
                        bssids.add(net.get('bssid', ''))
                        rssi.append(net.get('rssi', -100))
        row['wifi_n_unique']  = len(bssids)
        row['wifi_mean_rssi'] = np.mean(rssi) if rssi else np.nan
        row['wifi_max_rssi']  = np.max(rssi)  if rssi else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_ble(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        addrs = set()
        all_rssi, strong_addrs, night_addrs = [], set(), set()
        for ts, v in zip(grp['timestamp'], grp['m_ble']):
            arr = v.tolist() if hasattr(v, 'tolist') else v
            if isinstance(arr, list):
                for dev in arr:
                    if isinstance(dev, dict):
                        addr = dev.get('address', '')
                        rssi = dev.get('rssi', -100)
                        if isinstance(rssi, str):
                            try: rssi = int(rssi)
                            except: rssi = -100
                        addrs.add(addr)
                        all_rssi.append(rssi)
                        if rssi > -70:
                            strong_addrs.add(addr)
                        if ts.hour >= 22:
                            night_addrs.add(addr)
        row['ble_n_unique']         = len(addrs)
        row['ble_n_scans']          = len(grp)
        row['ble_mean_rssi']        = np.mean(all_rssi)        if all_rssi   else np.nan
        row['ble_strong_dev_count'] = len(strong_addrs)
        row['ble_night_unique']     = len(night_addrs)
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
                dm = {item[0]: item[1] for item in v if isinstance(item, list) and len(item) == 2}
                music_s.append(dm.get('Music', 0))
                speech_s.append(dm.get('Speech', 0))
                silence_s.append(dm.get('Silence', 0))
        night_speech = []
        for ts, v in zip(grp['timestamp'], grp['m_ambience']):
            if ts.hour >= 22 and isinstance(v, list):
                dm = {item[0]: item[1] for item in v if isinstance(item, list) and len(item) == 2}
                night_speech.append(dm.get('Speech', 0))
        row['amb_music_mean']   = np.mean(music_s)    if music_s    else np.nan
        row['amb_speech_mean']  = np.mean(speech_s)   if speech_s   else np.nan
        row['amb_silence_mean'] = np.mean(silence_s)  if silence_s  else np.nan
        row['amb_night_speech'] = np.mean(night_speech) if night_speech else np.nan
        row['amb_n_records']    = len(grp)
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
        slp = np.array(all_v)
        slp = slp[slp > 0] if len(slp) > 0 else slp
        for k, v in agg_stats(slp, 'slp_hr').items(): row[k] = v
        row['slp_hr_deep_ratio']  = (slp < 55).mean()            if len(slp) > 0 else np.nan
        row['slp_hr_awake_ratio'] = (slp > 75).mean()            if len(slp) > 0 else np.nan
        row['slp_hr_light_ratio'] = ((slp>=55)&(slp<=75)).mean() if len(slp) > 0 else np.nan
        row['slp_hr_rmssd']       = float(np.sqrt(np.nanmean(np.diff(slp)**2))) if len(slp) > 1 else np.nan
        row['slp_hr_n_records']   = len(grp)
        row['slp_hr_early_mean']  = safe_mean(sum([hour_vals[h] for h in range(3)],   []))
        row['slp_hr_late_mean']   = safe_mean(sum([hour_vals[h] for h in range(6, 9)],[]))
        row['slp_hr_mid_mean']    = safe_mean(sum([hour_vals[h] for h in range(3, 6)],[]))
        row['slp_hr_range']       = float(np.ptp(slp))    if len(slp) > 0 else np.nan
        row['slp_hr_median']      = float(np.median(slp)) if len(slp) > 0 else np.nan
        if len(slp) > 5:
            rolling = pd.Series(slp).rolling(5, min_periods=1).mean().values
            row['slp_hr_spike_count'] = int((np.abs(slp - rolling) > 15).sum())
        else: row['slp_hr_spike_count'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_pedo(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        row['slp_pedo_steps']     = morn['step'].sum()
        row['slp_pedo_active']    = (morn['step'] > 5).sum()
        row['slp_pedo_calories']  = morn['burned_calories'].sum()
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
            row['slp_act_still_ratio']  = (acts == 0).mean()
            row['slp_act_active_ratio'] = ((acts==7)|(acts==8)).mean()
            row['slp_act_n_records']    = len(acts)
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
            row['slp_screen_on']    = (sc > 0).sum()
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
            row['slp_wlight_mean']  = safe_mean(vals)
            row['slp_wlight_dark']  = (vals < 5).mean()   if len(vals) > 0 else np.nan
            row['slp_wlight_light'] = (vals > 100).mean() if len(vals) > 0 else np.nan
        else:
            row['slp_wlight_mean'] = np.nan; row['slp_wlight_dark'] = np.nan; row['slp_wlight_light'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def _build_subject_history(history_df, target):
    h = history_df[['subject_id','lifelog_date',target]].sort_values(['subject_id','lifelog_date']).reset_index(drop=True)
    hist = {}
    for sid, grp in h.groupby('subject_id'):
        hist[sid] = {'dates': grp['lifelog_date'].to_numpy(), 'labels': grp[target].to_numpy()}
    return hist

def _encode_from_history(history_map, query_df, windows):
    rows = []
    for sid, d in query_df[['subject_id','lifelog_date']].itertuples(index=False):
        if sid not in history_map:
            row = {'te_lag1': np.nan}
            for w in windows: row[f'te_enc{w}'] = np.nan
            rows.append(row); continue
        dates  = history_map[sid]['dates']
        labels = history_map[sid]['labels']
        k      = np.searchsorted(dates, d, side='left')
        past   = labels[:k]
        row    = {'te_lag1': past[-1] if len(past) > 0 else np.nan}
        for w in windows:
            row[f'te_enc{w}'] = np.nanmean(past[-w:]) if len(past) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=query_df.index)


def calibrate_probs(y_true, oof_prob, test_prob):
    oof_prob  = np.clip(oof_prob,  1e-7, 1-1e-7)
    test_prob = np.clip(test_prob, 1e-7, 1-1e-7)
    if CALIBRATION_METHOD == 'isotonic':
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(oof_prob, y_true)
        return np.clip(cal.transform(oof_prob), 1e-7, 1-1e-7), np.clip(cal.transform(test_prob), 1e-7, 1-1e-7)
    cal = LogisticRegression(solver='lbfgs', max_iter=1000)
    cal.fit(oof_prob.reshape(-1,1), y_true)
    return (np.clip(cal.predict_proba(oof_prob.reshape(-1,1))[:,1], 1e-7, 1-1e-7),
            np.clip(cal.predict_proba(test_prob.reshape(-1,1))[:,1], 1e-7, 1-1e-7))

def build_pseudo_public_mask(df, tail_frac):
    mask = pd.Series(False, index=df.index)
    for _, grp in df.sort_values(['subject_id','lifelog_date']).groupby('subject_id'):
        n = len(grp); tail_n = max(1, int(np.ceil(n * tail_frac)))
        mask.loc[grp.index[-tail_n:]] = True
    return mask.values

def build_feature_table(train_df, sub_df):
    all_keys = pd.concat([train_df[['subject_id','lifelog_date']],
                          sub_df[['subject_id','lifelog_date']]]).drop_duplicates().reset_index(drop=True)
    sleep_keys = pd.concat([train_df[['subject_id','sleep_date']],
                             sub_df[['subject_id','sleep_date']]]).drop_duplicates().reset_index(drop=True)

    print('Extracting daytime features...')
    feat_dfs = []
    for name, fn, col, prefix in [
        ('mActivity',extract_activity,None,None),('wPedo',extract_pedo,None,None),
        ('wHr',extract_hr,None,None),('mScreenStatus',extract_screen,None,None),
        ('mLight',extract_light,'m_light','mlight'),('wLight',extract_wlight,None,None),
        ('mACStatus',extract_ac,None,None),('mGps',extract_gps,None,None),
        ('mUsageStats',extract_usage,None,None),('mWifi',extract_wifi,None,None),
        ('mBle',extract_ble,None,None),('mAmbience',extract_ambience,None,None),
    ]:
        print(f'  {name}...')
        df = load_parquet(name)
        feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))
        del df; gc.collect()

    print('Extracting sleep-date features...')
    sleep_feat_dfs = []
    for name, fn in [('wHr',extract_sleep_hr),('wPedo',extract_sleep_pedo),
                     ('mActivity',extract_sleep_activity),('mScreenStatus',extract_sleep_screen),
                     ('wLight',extract_sleep_light)]:
        print(f'  sleep_morning: {name}...')
        df = load_parquet(name)
        sleep_feat_dfs.append(fn(df, sleep_keys))
        del df; gc.collect()

    sleep_feats = sleep_feat_dfs[0]
    for df in sleep_feat_dfs[1:]: sleep_feats = sleep_feats.merge(df, on=['subject_id','sleep_date'], how='outer')
    feat_all = feat_dfs[0]
    for df in feat_dfs[1:]: feat_all = feat_all.merge(df, on=['subject_id','lifelog_date'], how='outer')

    feat_all['dow']         = feat_all['lifelog_date'].dt.dayofweek
    feat_all['month']       = feat_all['lifelog_date'].dt.month
    feat_all['week']        = feat_all['lifelog_date'].dt.isocalendar().week.astype(int)
    feat_all['is_weekend']  = (feat_all['dow'] >= 5).astype(int)
    feat_all['subject_num'] = feat_all['subject_id'].str.extract(r'(\d+)').astype(int)
    feat_all['dow_sin']     = np.sin(2 * np.pi * feat_all['dow'] / 7)
    feat_all['dow_cos']     = np.cos(2 * np.pi * feat_all['dow'] / 7)
    feat_all['month_sin']   = np.sin(2 * np.pi * feat_all['month'] / 12)
    feat_all['month_cos']   = np.cos(2 * np.pi * feat_all['month'] / 12)

    if 'screen_on_total' in feat_all.columns and 'pedo_total_steps' in feat_all.columns:
        feat_all['screen_per_step'] = feat_all['screen_on_total'] / (feat_all['pedo_total_steps'] + 1)
    if 'pedo_total_calories' in feat_all.columns and 'usage_total_time' in feat_all.columns:
        feat_all['cal_per_usage'] = feat_all['pedo_total_calories'] / (feat_all['usage_total_time'] + 1)

    feat_all = feat_all.sort_values(['subject_id','lifelog_date']).reset_index(drop=True)

    roll_cols = [
        'pedo_total_steps','pedo_total_calories','pedo_total_distance',
        'screen_on_ratio','screen_night_on','screen_eve_ratio',
        'act_active_ratio','act_still_ratio','mlight_all_mean','wlight_all_mean',
        'gps_moving_ratio','usage_late_ratio','usage_eve_ratio','ac_presleep_charging',
        'hr_day_mean','hr_day_rmssd','hr_eve_mean','hr_presleep_mean',
        'ble_strong_dev_count','amb_night_speech',
    ]
    for col in roll_cols:
        if col not in feat_all.columns: continue
        g = feat_all.groupby('subject_id')[col]
        feat_all[f'{col}_lag1']  = g.shift(1)
        feat_all[f'{col}_lag2']  = g.shift(2)
        feat_all[f'{col}_diff1'] = feat_all[col] - feat_all[f'{col}_lag1']
        feat_all[f'{col}_ewma3'] = g.transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
        feat_all[f'{col}_ewma7'] = g.transform(lambda x: x.shift(1).ewm(span=7, adjust=False).mean())
        feat_all[f'{col}_roll3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        feat_all[f'{col}_roll7'] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        feat_all[f'{col}_roll14']= g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())

    train_full = train_df.merge(feat_all, on=['subject_id','lifelog_date'], how='left')
    train_full = train_full.merge(sleep_feats, on=['subject_id','sleep_date'], how='left')
    test_full  = sub_df[['subject_id','lifelog_date','sleep_date']].merge(feat_all, on=['subject_id','lifelog_date'], how='left')
    test_full  = test_full.merge(sleep_feats, on=['subject_id','sleep_date'], how='left')

    numeric_cols      = feat_all.select_dtypes(include=[np.number]).columns.tolist()
    exclude_from_norm = {'subject_num','dow','month','week','is_weekend','dow_sin','dow_cos','month_sin','month_cos'}
    norm_cols = [c for c in numeric_cols if c not in exclude_from_norm
                 and not any(s in c for s in ['lag','roll','ewma','diff'])]

    if not USE_TRAIN_SUBJ_NORM:
        for col in norm_cols:
            mu  = feat_all.groupby('subject_id')[col].transform('mean')
            sig = feat_all.groupby('subject_id')[col].transform('std').replace(0, np.nan)
            feat_all[f'{col}_subj_z'] = (feat_all[col] - mu) / sig
        train_full = train_df.merge(feat_all, on=['subject_id','lifelog_date'], how='left')
        train_full = train_full.merge(sleep_feats, on=['subject_id','sleep_date'], how='left')
        test_full  = sub_df[['subject_id','lifelog_date','sleep_date']].merge(feat_all, on=['subject_id','lifelog_date'], how='left')
        test_full  = test_full.merge(sleep_feats, on=['subject_id','sleep_date'], how='left')
    else:
        for col in norm_cols:
            tmp = train_full[['subject_id', col]].copy()
            subj_mu  = tmp.groupby('subject_id')[col].mean()
            subj_std = tmp.groupby('subject_id')[col].std().replace(0, np.nan)
            global_mu, global_std = tmp[col].mean(), tmp[col].std()
            train_full[f'{col}_subj_z'] = (train_full[col] - train_full['subject_id'].map(subj_mu)) / train_full['subject_id'].map(subj_std)
            test_full[f'{col}_subj_z']  = (test_full[col] - test_full['subject_id'].map(subj_mu).fillna(global_mu)) / test_full['subject_id'].map(subj_std).fillna(global_std)

    feature_cols = [c for c in train_full.columns if c not in ['subject_id','lifelog_date','sleep_date'] + TARGETS]
    print(f'Total features: {len(feature_cols)}')
    return train_full, test_full, feature_cols


def add_forward_features(train_full, test_full, feature_cols, windows=None):
    """
    같은 subject의 '미래 훈련 데이터'를 참조하는 forward-looking 피처.
    pool은 train rows만 사용 (test rows를 future reference로 쓰지 않음).
    interleaved block 구조에서: test S-block 이후 train T2-block이 존재하면
    test 행은 자신보다 날짜가 늦은 train 행을 참조할 수 있음.
    """
    if windows is None:
        windows = FWD_WINDOWS

    fwd_key_cols = [c for c in [
        'pedo_total_steps', 'pedo_total_calories',
        'hr_day_mean', 'hr_eve_mean', 'hr_presleep_mean',
        'screen_on_ratio', 'screen_night_on',
        'act_active_ratio', 'usage_late_ratio',
        'ac_presleep_charging', 'ble_strong_dev_count',
        'amb_night_speech', 'gps_moving_ratio',
    ] if c in train_full.columns]

    if not fwd_key_cols:
        print('  [FwdFeatures] No key cols found, skipping.')
        return train_full, test_full, feature_cols

    # pool: 훈련 행만 (augmentation 이전, 즉 원본 train_full)
    pool_by_subj = {}
    for subj, grp in train_full.groupby('subject_id'):
        pool_by_subj[subj] = (
            grp[['lifelog_date'] + fwd_key_cols]
            .sort_values('lifelog_date')
            .reset_index(drop=True)
        )

    new_col_names = [f'{c}_fwd{w}' for c in fwd_key_cols for w in windows]

    def compute_fwd(query_df):
        results = {nc: np.full(len(query_df), np.nan) for nc in new_col_names}
        for row_i, (_, row) in enumerate(query_df.iterrows()):
            subj = row['subject_id']
            d    = pd.Timestamp(row['lifelog_date'])
            if subj not in pool_by_subj:
                continue
            pool   = pool_by_subj[subj]
            future = pool[pool['lifelog_date'] > d]
            if len(future) == 0:
                continue
            for c in fwd_key_cols:
                for w in windows:
                    w_rows = future[future['lifelog_date'] <= d + pd.Timedelta(days=w)]
                    val = w_rows[c].mean() if len(w_rows) > 0 else np.nan
                    results[f'{c}_fwd{w}'][row_i] = val
        return pd.DataFrame(results, index=query_df.index)

    print(f'  [FwdFeatures] Computing forward features ({len(fwd_key_cols)} cols × {windows} days)...')
    fwd_train = compute_fwd(train_full.reset_index(drop=True))
    fwd_test  = compute_fwd(test_full.reset_index(drop=True))

    train_out = pd.concat([train_full.reset_index(drop=True), fwd_train.reset_index(drop=True)], axis=1)
    test_out  = pd.concat([test_full.reset_index(drop=True),  fwd_test.reset_index(drop=True)],  axis=1)
    new_features = feature_cols + new_col_names
    print(f'  [FwdFeatures] Added {len(new_col_names)} forward features → total {len(new_features)}')
    return train_out, test_out, new_features


def augment_with_pseudo_labels(train_full, test_full):
    """
    v23 test 예측에서 confidence 높은 행을 hard label로 훈련 데이터에 추가.
    - p > PSEUDO_CONFIDENCE → label=1
    - p < (1-PSEUDO_CONFIDENCE) → label=0
    - 그 외 → NaN (해당 target의 pseudo label로 쓰지 않음)
    반환: augmented train_full, n_orig (원본 행 수)
    """
    pseudo_path = OUTPUT_DIR / PSEUDO_SOURCE
    if not pseudo_path.exists():
        print(f'  [Pseudo] WARNING: {pseudo_path} not found. Skipping pseudo-labeling.')
        return train_full, len(train_full)

    n_orig     = len(train_full)
    pseudo_sub = pd.read_csv(pseudo_path)
    pseudo_sub['lifelog_date'] = pd.to_datetime(pseudo_sub['lifelog_date'])
    if 'sleep_date' in pseudo_sub.columns:
        pseudo_sub['sleep_date'] = pd.to_datetime(pseudo_sub['sleep_date'])

    # test_full 행들에 pseudo label 할당
    test_aug = test_full.copy().reset_index(drop=True)
    key_cols = ['subject_id', 'lifelog_date']
    pred_lookup = pseudo_sub.set_index(key_cols)

    for t in TARGETS:
        if t not in pred_lookup.columns:
            test_aug[t] = np.nan
            continue
        labels = np.full(len(test_aug), np.nan)
        for i, row in test_aug.iterrows():
            key = (row['subject_id'], row['lifelog_date'])
            if key in pred_lookup.index:
                raw = pred_lookup.loc[key, t]
                p   = float(raw.iloc[0]) if hasattr(raw, 'iloc') else float(raw)
                if p > PSEUDO_CONFIDENCE:
                    labels[i] = 1.0
                elif p < (1.0 - PSEUDO_CONFIDENCE):
                    labels[i] = 0.0
        test_aug[t] = labels

    # 어떤 target이라도 label이 있는 행만 추가
    has_any_label = test_aug[TARGETS].notna().any(axis=1)
    pseudo_rows   = test_aug[has_any_label].copy()
    n_pseudo      = len(pseudo_rows)
    n_labeled_rows = int(test_aug[TARGETS].notna().sum().sum())

    if n_pseudo == 0:
        print(f'  [Pseudo] No rows passed confidence threshold {PSEUDO_CONFIDENCE}. Skipping.')
        return train_full, n_orig

    augmented = pd.concat(
        [train_full.reset_index(drop=True), pseudo_rows.reset_index(drop=True)],
        ignore_index=True
    )
    print(f'  [Pseudo] {n_pseudo} pseudo rows added (conf={PSEUDO_CONFIDENCE}), '
          f'{n_labeled_rows} target labels total. n_orig={n_orig} → n_total={len(augmented)}')
    return augmented, n_orig


def compute_stability_scores(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    importances = []
    for tr_idx, _ in skf.split(X, y):
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1,
                                n_jobs=-1, random_state=42)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        imp = m.feature_importances_.astype(float)
        imp /= (imp.sum() + 1e-10)
        importances.append(imp)
    importances = np.array(importances)
    nonzero_rate = (importances > 0).mean(axis=0)
    mean_imp     = importances.mean(axis=0)
    return nonzero_rate, mean_imp


def train_and_predict(train_full, test_full, feature_cols, n_orig):
    """
    n_orig: 원본 훈련 행 수. train_full[n_orig:] 는 pseudo 행.
    - stability filter / Optuna / OOF 평가: 원본 행만 사용
    - L0 fold 학습: 각 fold의 train set에 pseudo 행 concat
    - val set: 항상 원본 행만
    """
    n_total      = len(train_full)
    X_train_base = train_full[feature_cols].copy()
    X_test_base  = test_full[feature_cols].copy()

    meta_params     = {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000}
    seeds           = [42, 1234, 9999, 7, 314, 2025, 777, 555, 2077, 1337, 99, 1111]
    n_folds         = 5
    n_optuna_trials = 50
    te_windows      = [3, 7, 14, 21]

    oof_preds  = np.zeros((n_orig, len(TARGETS)))
    test_preds = np.zeros((len(X_test_base), len(TARGETS)))

    for ti, target in enumerate(TARGETS):
        y_full = train_full[target].values          # NaN for uncertain pseudo rows
        y_orig = y_full[:n_orig]

        # pseudo 행 중 이 target에 label이 있는 것의 절대 인덱스
        if n_total > n_orig:
            pseudo_mask_t   = ~np.isnan(y_full[n_orig:])
            pseudo_valid_idx = np.where(pseudo_mask_t)[0] + n_orig
        else:
            pseudo_valid_idx = np.array([], dtype=int)

        print(f'\n{"="*40}\n=== Target: {target} | pos_rate: {y_orig.mean():.3f} | '
              f'pseudo_rows: {len(pseudo_valid_idx)} ===\n{"="*40}')

        # --- Stability filter (원본 행만) ---
        print(f'  [Stability] Computing feature stability scores...')
        X_orig_base = X_train_base.iloc[:n_orig]
        nonzero_rate, mean_imp = compute_stability_scores(X_orig_base, y_orig)
        stable_idx  = np.where(nonzero_rate >= STABILITY_THRESHOLD)[0]
        stable_idx  = stable_idx[np.argsort(mean_imp[stable_idx])[::-1]][:MAX_STABLE_FEATURES]
        stable_cols = [feature_cols[i] for i in stable_idx]
        if len(stable_cols) == 0:
            stable_cols = feature_cols
        print(f'  [Stability] {len(feature_cols)} → {len(stable_cols)} features selected')

        X_orig_use      = X_orig_base[stable_cols]          # n_orig × stable
        X_train_use_all = X_train_base[stable_cols]         # n_total × stable (incl. pseudo)
        X_test_use      = X_test_base[stable_cols]

        # --- Optuna tuning (원본 행, 3-fold StratifiedKFold) ---
        print(f'  [Optuna] Searching Golden Parameters ({n_optuna_trials} trials/model)...')
        tune_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def objective_lgb(trial):
            params = {
                'objective':'binary','metric':'binary_logloss','boosting_type':'gbdt',
                'n_estimators':300,'verbose':-1,'n_jobs':-1,'random_state':42,
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves':       trial.suggest_int('num_leaves', 15, 63),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'min_child_samples':trial.suggest_int('min_child_samples', 10, 50),
            }
            if HAS_CUDA: params.update({'device':'gpu'})
            losses = []
            for tr_idx, val_idx in tune_skf.split(X_orig_use, y_orig):
                m = lgb.LGBMClassifier(**params)
                try: m.fit(X_orig_use.iloc[tr_idx], y_orig[tr_idx])
                except:
                    params['device'] = 'cpu'; m = lgb.LGBMClassifier(**params)
                    m.fit(X_orig_use.iloc[tr_idx], y_orig[tr_idx])
                losses.append(log_loss(y_orig[val_idx], m.predict_proba(X_orig_use.iloc[val_idx])[:,1]))
            return np.mean(losses)

        def objective_xgb(trial):
            params = {
                'objective':'binary:logistic','eval_metric':'logloss',
                'n_estimators':300,'n_jobs':-1,'random_state':42,
                'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth':       trial.suggest_int('max_depth', 3, 8),
                'subsample':       trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 0.9),
            }
            if HAS_CUDA: params.update({'tree_method':'hist','device':'cuda'})
            losses = []
            for tr_idx, val_idx in tune_skf.split(X_orig_use, y_orig):
                m = xgb.XGBClassifier(**params)
                m.fit(X_orig_use.iloc[tr_idx], y_orig[tr_idx], verbose=False)
                losses.append(log_loss(y_orig[val_idx], m.predict_proba(X_orig_use.iloc[val_idx])[:,1]))
            return np.mean(losses)

        def objective_cat(trial):
            params = {
                'loss_function':'Logloss','iterations':300,'verbose':False,
                'thread_count':-1,'random_seed':42,
                'learning_rate':trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth':        trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg':  trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            }
            if HAS_CUDA: params.update({'task_type':'GPU'})
            losses = []
            for tr_idx, val_idx in tune_skf.split(X_orig_use, y_orig):
                m = CatBoostClassifier(**params)
                m.fit(X_orig_use.iloc[tr_idx], y_orig[tr_idx])
                losses.append(log_loss(y_orig[val_idx], m.predict_proba(X_orig_use.iloc[val_idx])[:,1]))
            return np.mean(losses)

        study_lgb = optuna.create_study(direction='minimize'); study_lgb.optimize(objective_lgb, n_trials=n_optuna_trials)
        study_xgb = optuna.create_study(direction='minimize'); study_xgb.optimize(objective_xgb, n_trials=n_optuna_trials)
        study_cat = optuna.create_study(direction='minimize'); study_cat.optimize(objective_cat, n_trials=n_optuna_trials)
        print(f'    LGBM best: {study_lgb.best_value:.4f}  XGB: {study_xgb.best_value:.4f}  CAT: {study_cat.best_value:.4f}')

        best_lgb = {**study_lgb.best_params, 'objective':'binary','metric':'binary_logloss',
                    'boosting_type':'gbdt','n_estimators':2000,'verbose':-1,'n_jobs':-1}
        best_xgb = {**study_xgb.best_params, 'objective':'binary:logistic','eval_metric':'logloss',
                    'n_estimators':2000,'n_jobs':-1,'early_stopping_rounds':100}
        best_cat = {**study_cat.best_params, 'loss_function':'Logloss','iterations':2000,
                    'verbose':False,'thread_count':-1}
        if HAS_CUDA:
            best_lgb.update({'device':'gpu','gpu_platform_id':0,'gpu_device_id':0})
            best_xgb.update({'tree_method':'hist','device':'cuda'})
            best_cat.update({'task_type':'GPU'})
        else:
            best_lgb.update({'device':'cpu'})

        # --- L0: 12-seed StratifiedKFold ensemble (pseudo augmentation 포함) ---
        print(f'  [L0] Training {len(seeds)}-seed ensemble (pseudo aug={len(pseudo_valid_idx)} rows)...')
        oof_lgb  = np.zeros(n_orig)
        oof_xgb  = np.zeros(n_orig)
        oof_cat  = np.zeros(n_orig)
        test_lgb = np.zeros(len(X_test_base))
        test_xgb = np.zeros(len(X_test_base))
        test_cat = np.zeros(len(X_test_base))

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            s_oof_lgb  = np.zeros(n_orig)
            s_oof_xgb  = np.zeros(n_orig)
            s_oof_cat  = np.zeros(n_orig)
            s_test_lgb = np.zeros(len(X_test_base))
            s_test_xgb = np.zeros(len(X_test_base))
            s_test_cat = np.zeros(len(X_test_base))

            for tr_idx, val_idx in skf.split(X_orig_use, y_orig):
                # TE history: 원본 훈련 fold rows만 (pseudo 포함 금지)
                hist_df    = train_full.iloc[tr_idx][['subject_id','lifelog_date',target]].copy()
                hmap       = _build_subject_history(hist_df, target)

                tr_query   = train_full.iloc[tr_idx][['subject_id','lifelog_date']].copy()
                val_query  = train_full.iloc[val_idx][['subject_id','lifelog_date']].copy()
                test_query = test_full[['subject_id','lifelog_date']].copy()

                tr_te   = _encode_from_history(hmap, tr_query.reset_index(drop=True),   te_windows)
                val_te  = _encode_from_history(hmap, val_query.reset_index(drop=True),  te_windows)
                test_te = _encode_from_history(hmap, test_query.reset_index(drop=True), te_windows)

                X_tr_orig = pd.concat([X_train_use_all.iloc[tr_idx].reset_index(drop=True),
                                       tr_te.reset_index(drop=True)], axis=1)
                X_val     = pd.concat([X_train_use_all.iloc[val_idx].reset_index(drop=True),
                                       val_te.reset_index(drop=True)], axis=1)
                X_te      = pd.concat([X_test_use.reset_index(drop=True),
                                       test_te.reset_index(drop=True)], axis=1)

                y_tr = y_orig[tr_idx]

                # pseudo 행 TE: fold의 훈련 history 사용 (test 행이므로 leakage 없음)
                if len(pseudo_valid_idx) > 0:
                    pseudo_query = train_full.iloc[pseudo_valid_idx][['subject_id','lifelog_date']].copy()
                    pseudo_te    = _encode_from_history(hmap, pseudo_query.reset_index(drop=True), te_windows)
                    X_tr_pseudo  = pd.concat([X_train_use_all.iloc[pseudo_valid_idx].reset_index(drop=True),
                                              pseudo_te.reset_index(drop=True)], axis=1)
                    X_tr   = pd.concat([X_tr_orig, X_tr_pseudo], ignore_index=True)
                    y_tr   = np.concatenate([y_orig[tr_idx], y_full[pseudo_valid_idx]])
                else:
                    X_tr = X_tr_orig

                m_lgb = lgb.LGBMClassifier(**{**best_lgb, 'random_state': seed})
                m_xgb = xgb.XGBClassifier(**{**best_xgb, 'random_state': seed})
                m_cat = CatBoostClassifier(**{**best_cat, 'random_seed': seed})

                try:
                    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_orig[val_idx])],
                              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
                except Exception:
                    cp = dict(best_lgb); cp['device'] = 'cpu'
                    m_lgb = lgb.LGBMClassifier(**{**cp, 'random_state': seed})
                    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_orig[val_idx])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])

                m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_orig[val_idx])], verbose=False)
                m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_orig[val_idx]),
                          early_stopping_rounds=100, verbose=False)

                s_oof_lgb[val_idx]  = m_lgb.predict_proba(X_val)[:, 1]
                s_oof_xgb[val_idx]  = m_xgb.predict_proba(X_val)[:, 1]
                s_oof_cat[val_idx]  = m_cat.predict_proba(X_val)[:, 1]
                s_test_lgb += m_lgb.predict_proba(X_te)[:, 1] / n_folds
                s_test_xgb += m_xgb.predict_proba(X_te)[:, 1] / n_folds
                s_test_cat += m_cat.predict_proba(X_te)[:, 1] / n_folds

            oof_lgb  += s_oof_lgb  / len(seeds)
            oof_xgb  += s_oof_xgb  / len(seeds)
            oof_cat  += s_oof_cat  / len(seeds)
            test_lgb += s_test_lgb / len(seeds)
            test_xgb += s_test_xgb / len(seeds)
            test_cat += s_test_cat / len(seeds)

        print(f'  [L0] LGBM={log_loss(y_orig,oof_lgb):.4f}  XGB={log_loss(y_orig,oof_xgb):.4f}  CAT={log_loss(y_orig,oof_cat):.4f}')

        # --- L1: meta stacking (원본 행만) ---
        print('  [L1] Meta stacking...')
        X_meta_tr = np.column_stack([oof_lgb,  oof_xgb,  oof_cat])
        X_meta_te = np.column_stack([test_lgb, test_xgb, test_cat])
        meta_oof  = np.zeros(n_orig)
        meta_test = np.zeros(len(X_test_base))
        meta_skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for meta_tr_idx, meta_val_idx in meta_skf.split(X_meta_tr, y_orig):
            mm = LogisticRegression(**meta_params)
            mm.fit(X_meta_tr[meta_tr_idx], y_orig[meta_tr_idx])
            meta_oof[meta_val_idx]  = mm.predict_proba(X_meta_tr[meta_val_idx])[:, 1]
            meta_test              += mm.predict_proba(X_meta_te)[:, 1] / n_folds

        target_oof  = meta_oof
        target_test = meta_test
        if USE_CALIBRATION: target_oof, target_test = calibrate_probs(y_orig, target_oof, target_test)

        oof_preds[:, ti]  = target_oof
        test_preds[:, ti] = target_test
        print(f'  [L1] Stacked OOF [{target}]: {log_loss(y_orig, target_oof):.4f}')

    return oof_preds, test_preds


def main():
    ensure_dirs()
    log_f = open(RUN_LOG_PATH, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG}...')
    print(f'  PSEUDO_CONFIDENCE={PSEUDO_CONFIDENCE}  FWD_WINDOWS={FWD_WINDOWS}  MAX_STABLE_FEATURES={MAX_STABLE_FEATURES}')

    train_df = pd.read_csv(TRAIN_PATH)
    sub_df   = pd.read_csv(SUB_PATH)
    for df in [train_df, sub_df]:
        df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
        df['sleep_date']   = pd.to_datetime(df['sleep_date'])

    train_full, test_full, feature_cols = build_feature_table(train_df, sub_df)

    print('\n[ForwardFeatures] Building forward-looking features...')
    train_full, test_full, feature_cols = add_forward_features(train_full, test_full, feature_cols)

    print('\n[PseudoLabeling] Augmenting training set with pseudo labels...')
    train_full, n_orig = augment_with_pseudo_labels(train_full, test_full)

    oof_preds, test_preds = train_and_predict(train_full, test_full, feature_cols, n_orig)

    # --- OOF 평가 (원본 행만) ---
    train_orig = train_full.iloc[:n_orig]
    per_target = {t: log_loss(train_orig[t].values, oof_preds[:, i])
                  for i, t in enumerate(TARGETS)}
    oof_total = float(np.mean(list(per_target.values())))

    pseudo_mask = build_pseudo_public_mask(
        train_orig[['subject_id','lifelog_date']], PSEUDO_PUBLIC_TAIL_FRAC)
    pseudo_per_target = {t: log_loss(train_orig.loc[pseudo_mask, t].values, oof_preds[pseudo_mask, i])
                         for i, t in enumerate(TARGETS)}
    pseudo_oof_total = float(np.mean(list(pseudo_per_target.values())))

    print(f'\n{"="*55}')
    print(f'{EXP_TAG} Total OOF:          {oof_total:.4f}')
    print(f'{EXP_TAG} Pseudo-public OOF:  {pseudo_oof_total:.4f}')
    print(f'{"="*55}')
    for t in TARGETS:
        print(f'  {t}: OOF={per_target[t]:.4f}  pseudo={pseudo_per_target[t]:.4f}')

    oof_df = train_orig[['subject_id','sleep_date','lifelog_date']].copy()
    for i, t in enumerate(TARGETS):
        oof_df[t] = oof_preds[:, i]
    oof_path = OOF_DIR / f'oof_{EXP_TAG}.csv'
    oof_df.to_csv(oof_path, index=False)
    print(f'oof saved: {oof_path}')

    submission = sub_df[['subject_id','sleep_date','lifelog_date']].copy()
    for i, t in enumerate(TARGETS):
        submission[t] = test_preds[:, i].clip(0.02, 0.98)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f'submission saved: {OUTPUT_PATH}')
    log_f.close()

if __name__ == '__main__':
    main()

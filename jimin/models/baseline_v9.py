import warnings
warnings.filterwarnings('ignore')

import os
import gc
import json
import datetime
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


DATA_DIR = './ch2025_data_items'
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
EXP_TAG = '_all_features_no_lag_pseudolabel'
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR = OUTPUTS_DIR / 'submissions'
REPORT_DIR = OUTPUTS_DIR / 'report'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'

OUTPUT_PATH = str(OUTPUT_DIR / 'submission_v9.csv')
REPORT_PATH = str(REPORT_DIR / 'report_v9.txt')
SUMMARY_PATH = str(SUMMARY_DIR / 'summary_v9.json')
OOF_PATH = str(OOF_DIR / 'oof_v9.csv')
TEST_PREDS_PATH = str(REPORT_DIR / 'test_preds_v9.csv')
RUN_LOG_PATH = str(LOG_DIR / 'run_v9.log')


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


_stdout = sys.stdout
_stderr = sys.stderr
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
_run_log_handle = open(RUN_LOG_PATH, 'w', encoding='utf-8')
sys.stdout = Tee(_stdout, _run_log_handle)
sys.stderr = Tee(_stderr, _run_log_handle)

print('Loading raw data...')
train_df = pd.read_csv('ch2026_metrics_train.csv')
sub_df   = pd.read_csv('ch2026_submission_sample.csv')
train_df['lifelog_date'] = pd.to_datetime(train_df['lifelog_date'])
sub_df['lifelog_date']   = pd.to_datetime(sub_df['lifelog_date'])
train_df['sleep_date']   = pd.to_datetime(train_df['sleep_date'])
sub_df['sleep_date']     = pd.to_datetime(sub_df['sleep_date'])

all_keys = pd.concat([
    train_df[['subject_id','lifelog_date']],
    sub_df[['subject_id','lifelog_date']]
]).drop_duplicates().reset_index(drop=True)

sleep_keys = pd.concat([
    train_df[['subject_id','sleep_date']],
    sub_df[['subject_id','sleep_date']]
]).drop_duplicates().reset_index(drop=True)

print(f'train: {train_df.shape}, test: {sub_df.shape}')


def write_report(report_data, path=REPORT_PATH):
    lines = []
    lines.append('=' * 80)
    lines.append('Baseline v9 run report')
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f'Experiment tag: {EXP_TAG}')
    lines.append(f'Run log: {RUN_LOG_PATH}')
    lines.append(f'Submission: {OUTPUT_PATH}')
    lines.append(f'OOF: {OOF_PATH}')
    lines.append(f'Test preds: {TEST_PREDS_PATH}')
    lines.append('')
    lines.append('[Summary]')
    lines.append(f"  Total OOF: {report_data['avg_oof']:.4f}")
    lines.append(f"  Feature count: {report_data['n_features']}")
    lines.append(f"  Train samples: {report_data['n_train']}")
    lines.append(f"  Test samples: {report_data['n_test']}")
    lines.append('')
    lines.append('[Per target]')
    for target in TARGETS:
        lines.append(f"  {target}: {report_data['per_target_oof'][target]:.4f}")

    text = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('\n' + text)
    print(f'\n[리포트 저장] {path}')

def get_our_refined_sleep_features():
    print("\n[Step 1] 우리 팀의 정교한 수면 피처(v9_refined_sleep) 추출 시작...")
    
    # 데이터 로드
    screen = pd.read_parquet(f'{DATA_DIR}/ch2025_mScreenStatus.parquet')
    activity = pd.read_parquet(f'{DATA_DIR}/ch2025_mActivity.parquet')
    hr = pd.read_parquet(f'{DATA_DIR}/ch2025_wHr.parquet')

    def to_list(x): return x if isinstance(x, (list, np.ndarray)) else []
    def safe_mean(x): return np.mean(x) if len(x) > 0 else np.nan
    def safe_std(x): return np.std(x) if len(x) > 0 else np.nan
    def safe_min(x): return np.min(x) if len(x) > 0 else np.nan

    # 분 단위 집계
    screen['minute'] = screen['timestamp'].dt.floor('min')
    activity['minute'] = activity['timestamp'].dt.floor('min')
    hr['minute'] = hr['timestamp'].dt.floor('min')

    # HR 평균 계산
    hr['hr_mean_min'] = hr['heart_rate'].apply(lambda x: safe_mean(to_list(x)))
    hr_min = hr[['subject_id','minute','hr_mean_min']].drop_duplicates(subset=['subject_id','minute'])

    # 병합 및 수면 추정 로직
    merged = screen[['subject_id','minute','m_screen_use']].merge(
        activity[['subject_id','minute','m_activity']], on=['subject_id','minute'], how='outer')
    merged = merged.merge(hr_min, on=['subject_id','minute'], how='left')
    merged = merged.sort_values(['subject_id','minute']).reset_index(drop=True)

    subj_hr_means = merged.groupby('subject_id')['hr_mean_min'].mean().to_dict()
    merged['subj_hr_mean'] = merged['subject_id'].map(subj_hr_means)
    merged['screen_off'] = (merged['m_screen_use'].fillna(1) == 0).astype(int)
    merged['low_act'] = merged['m_activity'].isin([0, 3]).astype(int)
    merged['low_hr'] = (merged['hr_mean_min'] < merged['subj_hr_mean'] * 0.9).astype(int)
    merged['is_sleep'] = (merged['screen_off'] == 1) & ((merged['low_act'] == 1) | (merged['low_hr'] == 1))
    
    # 날짜 정규화 (병합 키: lifelog_date)
    merged['lifelog_date'] = merged['minute'].dt.normalize()

    rows = []
    for (subj, date), grp in merged.groupby(['subject_id','lifelog_date']):
        sleep_mask = grp['is_sleep'].values
        # 최장 연속 수면 구간 계산 로직 (v8과 동일)
        best_len, best_start, best_end = 0, None, None
        seg_start = None
        for i, s in enumerate(sleep_mask):
            if s == 1:
                if seg_start is None: seg_start = i
            else:
                if seg_start is not None:
                    if (i - seg_start) > best_len: best_len, best_start, best_end = i - seg_start, seg_start, i
                    seg_start = None
        if seg_start is not None and (len(sleep_mask) - seg_start) > best_len:
            best_len, best_start, best_end = len(sleep_mask) - seg_start, seg_start, len(sleep_mask)

        if best_len >= 30 and best_start is not None:
            seg = grp.iloc[best_start:best_end]
            hr_in_sleep = seg['hr_mean_min'].dropna().tolist()
            rows.append({
                'subject_id': subj, 'lifelog_date': date,
                'v9_refined_sleep_dur': best_len,
                'v9_refined_sleep_hr_mean': safe_mean(hr_in_sleep),
                'v9_refined_sleep_ratio': sleep_mask.mean()
            })
        else:
            rows.append({
                'subject_id': subj, 'lifelog_date': date,
                'v9_refined_sleep_dur': 0, 'v9_refined_sleep_hr_mean': np.nan, 'v9_refined_sleep_ratio': sleep_mask.mean()
            })
    
    del merged, screen, activity, hr; gc.collect()
    return pd.DataFrame(rows)

def agg_stats(vals, prefix):
    if len(vals) == 0:
        return {f'{prefix}_mean': np.nan, f'{prefix}_std': np.nan,
                f'{prefix}_min': np.nan, f'{prefix}_max': np.nan,
                f'{prefix}_median': np.nan, f'{prefix}_q25': np.nan,
                f'{prefix}_q75': np.nan}
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
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_{name}.parquet')
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
        row['act_still_ratio']  = (acts == 0).mean()
        row['act_n_records']    = len(acts)
        for seg, mask in [('morn', (h>=6)&(h<12)), ('aftn', (h>=12)&(h<18)),
                          ('eve',  (h>=18)&(h<22)), ('night', (h>=22)|(h<6))]:
            s_acts = acts[mask]
            row[f'act_{seg}_active'] = ((s_acts==7)|(s_acts==8)).mean() if len(s_acts)>0 else np.nan
            row[f'act_{seg}_still']  = (s_acts==0).mean() if len(s_acts)>0 else np.nan
        pre = acts[(h >= 22) & (h < 24)]
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
    # 낮시간 HR은 v5에서 NaN — 수면 HR(extract_sleep_hr)만 사용
    return pd.DataFrame({'subject_id': keys['subject_id'], 'lifelog_date': keys['lifelog_date']})

def extract_screen(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        sc = grp['m_screen_use'].values
        h  = grp['timestamp'].dt.hour.values
        row['screen_on_total']   = (sc > 0).sum()
        row['screen_on_ratio']   = (sc > 0).mean()
        row['screen_unlock_cnt'] = ((sc[1:] > sc[:-1])).sum()
        for seg, mask in [('night', (h>=22)|(h<2)), ('eve', (h>=20)&(h<=23)),
                          ('presleep', (h>=22)&(h<24))]:
            s_sc = sc[mask]
            row[f'screen_{seg}_on']    = (s_sc > 0).sum()
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
            sv = grp.loc[h.between(lo, hi-1), col].dropna().values
            row[f'{prefix}_{seg}_mean'] = safe_mean(sv)
        row[f'{prefix}_dark_ratio']   = (vals < 10).mean() if len(vals) > 0 else np.nan
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
        for seg, mask in [('eve', (h>=21)&(h<=23)), ('night', (h>=22)|(h<4)),
                          ('presleep', (h>=22)&(h<24))]:
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
        row['gps_mean_speed']   = np.nanmean(speeds) if len(speeds) > 0 else np.nan
        row['gps_max_speed']    = np.nanmax(speeds)  if len(speeds) > 0 else np.nan
        row['gps_moving_ratio'] = (speeds > 0.5).mean() if len(speeds) > 0 else np.nan
        row['gps_lat_std']      = np.nanstd(lats) if len(lats) > 0 else np.nan
        row['gps_lon_std']      = np.nanstd(lons) if len(lons) > 0 else np.nan
        if len(lats) > 1:
            dlat = np.diff(lats); dlon = np.diff(lons)
            row['gps_total_disp'] = float(np.sum(np.sqrt(dlat**2 + dlon**2)))
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
                        total_time += t; n_apps += 1
                        if ts.hour >= 22 or ts.hour < 2:
                            late_time += t
                        if ts.hour >= 18:
                            eve_time += t
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
        all_bssids, rssi_vals = set(), []
        for v in grp['m_wifi']:
            if isinstance(v, list):
                for net in v:
                    if isinstance(net, dict):
                        all_bssids.add(net.get('bssid',''))
                        rssi_vals.append(net.get('rssi', -100))
        row['wifi_n_unique']  = len(all_bssids)
        row['wifi_mean_rssi'] = np.mean(rssi_vals) if rssi_vals else np.nan
        row['wifi_max_rssi']  = np.max(rssi_vals)  if rssi_vals else np.nan
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
                        addrs.add(dev.get('address',''))
        row['ble_n_unique'] = len(addrs)
        row['ble_n_scans']  = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_wlight(df_raw, keys):
    return extract_light(df_raw, 'w_light', 'wlight', keys)

def extract_ambience(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        music_s, speech_s, silence_s = [], [], []
        for v in grp['m_ambience']:
            if isinstance(v, list):
                d_map = {item[0]: item[1] for item in v if isinstance(item, list) and len(item)==2}
                music_s.append(d_map.get('Music', 0))
                speech_s.append(d_map.get('Speech', 0))
                silence_s.append(d_map.get('Silence', 0))
        row['amb_music_mean']   = np.mean(music_s)   if music_s else np.nan
        row['amb_speech_mean']  = np.mean(speech_s)  if speech_s else np.nan
        row['amb_silence_mean'] = np.mean(silence_s) if silence_s else np.nan
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
            try:
                arr = np.asarray(v, dtype=float).ravel()
                arr = arr[arr > 0]
            except Exception:
                arr = np.array([])
            all_v.extend(arr.tolist())
            hour_vals[ts.hour].extend(arr.tolist())
        sleep_hrs = np.array(all_v)
        sleep_hrs = sleep_hrs[sleep_hrs > 0] if len(sleep_hrs) > 0 else sleep_hrs
        for k, v in agg_stats(sleep_hrs, 'slp_hr').items():
            row[k] = v
        row['slp_hr_deep_ratio']  = (sleep_hrs < 55).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_awake_ratio'] = (sleep_hrs > 75).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_light_ratio'] = ((sleep_hrs >= 55) & (sleep_hrs <= 75)).mean() if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 1:
            diffs = np.diff(sleep_hrs)
            row['slp_hr_rmssd'] = float(np.sqrt(np.nanmean(diffs**2)))
        else:
            row['slp_hr_rmssd'] = np.nan
        row['slp_hr_n_records']  = len(grp)
        row['slp_hr_early_mean'] = safe_mean(sum([hour_vals[h] for h in range(3)], []))
        row['slp_hr_late_mean']  = safe_mean(sum([hour_vals[h] for h in range(6, 9)], []))
        row['slp_hr_mid_mean']   = safe_mean(sum([hour_vals[h] for h in range(3, 6)], []))
        row['slp_hr_range']      = float(np.ptp(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_median']     = float(np.median(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 5:
            rolling = pd.Series(sleep_hrs).rolling(5, min_periods=1).mean().values
            row['slp_hr_spike_count'] = int((np.abs(sleep_hrs - rolling) > 15).sum())
        else:
            row['slp_hr_spike_count'] = np.nan
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
            row['slp_act_active_ratio'] = ((acts == 7) | (acts == 8)).mean()
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
            row['slp_screen_on'] = row['slp_screen_ratio'] = np.nan
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
            row['slp_wlight_dark']  = (vals < 5).mean() if len(vals) > 0 else np.nan
            row['slp_wlight_light'] = (vals > 100).mean() if len(vals) > 0 else np.nan
        else:
            row['slp_wlight_mean'] = row['slp_wlight_dark'] = row['slp_wlight_light'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

print('Extracting features...')
feat_dfs = []
for name, fn, col, prefix in [
    ('mActivity',    extract_activity,  None, None),
    ('wPedo',        extract_pedo,      None, None),
    ('wHr',          extract_hr,        None, None),
    ('mScreenStatus',extract_screen,    None, None),
    ('mLight',       extract_light,     'm_light',  'mlight'),
    ('wLight',       extract_wlight,    None, None),
    ('mACStatus',    extract_ac,        None, None),
    ('mGps',         extract_gps,       None, None),
    ('mUsageStats',  extract_usage,     None, None),
    ('mWifi',        extract_wifi,      None, None),
    ('mBle',         extract_ble,       None, None),
    ('mAmbience',    extract_ambience,  None, None),
]:
    print(f'  {name}...')
    df = load_parquet(name)
    feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))

sleep_feat_dfs = []
for name, fn in [
    ('wHr',          extract_sleep_hr),
    ('wPedo',        extract_sleep_pedo),
    ('mActivity',    extract_sleep_activity),
    ('mScreenStatus',extract_sleep_screen),
    ('wLight',       extract_sleep_light),
]:
    print(f'  sleep_morning: {name}...')
    df = load_parquet(name)
    sleep_feat_dfs.append(fn(df, sleep_keys))

sleep_feats = sleep_feat_dfs[0]
for df in sleep_feat_dfs[1:]:
    sleep_feats = sleep_feats.merge(df, on=['subject_id','sleep_date'], how='outer')

feat_all = feat_dfs[0]
for df in feat_dfs[1:]:
    feat_all = feat_all.merge(df, on=['subject_id','lifelog_date'], how='outer')

print(f'feat_all: {feat_all.shape}, sleep_feats: {sleep_feats.shape}')

# Time features
feat_all['dow']         = feat_all['lifelog_date'].dt.dayofweek
feat_all['month']       = feat_all['lifelog_date'].dt.month
feat_all['week']        = feat_all['lifelog_date'].dt.isocalendar().week.astype(int)
feat_all['is_weekend']  = (feat_all['dow'] >= 5).astype(int)
feat_all['subject_num'] = feat_all['subject_id'].str.extract(r'(\d+)').astype(int)
feat_all = feat_all.sort_values(['subject_id','lifelog_date']).reset_index(drop=True)

# Lag / rolling
roll_cols = [
    'pedo_total_steps','pedo_total_calories','pedo_total_distance',
    'screen_on_ratio','screen_night_on','screen_eve_ratio',
    'act_active_ratio','act_still_ratio',
    'mlight_all_mean','wlight_all_mean',
    'gps_moving_ratio','usage_late_ratio','usage_eve_ratio',
    'ac_presleep_charging',
]
for col in roll_cols:
    if col not in feat_all.columns:
        continue
    g = feat_all.groupby('subject_id')[col]
    feat_all[f'{col}_lag1']  = g.shift(1)
    feat_all[f'{col}_lag2']  = g.shift(2)
    feat_all[f'{col}_roll3']  = g.transform(lambda x: x.shift(1).rolling(3,  min_periods=1).mean())
    feat_all[f'{col}_roll7']  = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    feat_all[f'{col}_roll14'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())

# Subject z-score (transductive: train+test 기반)
numeric_cols = feat_all.select_dtypes(include=[np.number]).columns.tolist()
exclude_from_norm = {'subject_num','dow','month','week','is_weekend'}
norm_cols = [c for c in numeric_cols
             if c not in exclude_from_norm and 'lag' not in c and 'roll' not in c]
for col in norm_cols:
    mu  = feat_all.groupby('subject_id')[col].transform('mean')
    sig = feat_all.groupby('subject_id')[col].transform('std').replace(0, np.nan)
    feat_all[f'{col}_subj_z'] = (feat_all[col] - mu) / sig

# Merge
train_full = train_df.merge(feat_all, on=['subject_id','lifelog_date'], how='left')
train_full = train_full.merge(sleep_feats, on=['subject_id','sleep_date'], how='left')
test_full  = sub_df[['subject_id','lifelog_date','sleep_date']].merge(
    feat_all, on=['subject_id','lifelog_date'], how='left')
test_full  = test_full.merge(sleep_feats, on=['subject_id','sleep_date'], how='left')

# Target encoding (shift(1) — leakage 방지)
all_with_labels = pd.concat([
    train_full[['subject_id','lifelog_date'] + TARGETS],
    test_full[['subject_id','lifelog_date']].assign(**{t: np.nan for t in TARGETS})
], ignore_index=True).sort_values(['subject_id','lifelog_date'])

enc_cols = []
for t in TARGETS:
    for w in [3, 7, 14, 21]:
        col = f'{t}_enc{w}'
        all_with_labels[col] = all_with_labels.groupby('subject_id')[t].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        enc_cols.append(col)
    col_lag = f'{t}_lag1'
    all_with_labels[col_lag] = all_with_labels.groupby('subject_id')[t].shift(1)
    enc_cols.append(col_lag)

enc_df = all_with_labels[['subject_id','lifelog_date'] + enc_cols]
train_full = train_full.merge(enc_df, on=['subject_id','lifelog_date'], how='left')
test_full  = test_full.merge(enc_df, on=['subject_id','lifelog_date'], how='left')

df_our_sleep = get_our_refined_sleep_features()
train_full = train_full.merge(df_our_sleep, on=['subject_id', 'lifelog_date'], how='left')
test_full = test_full.merge(df_our_sleep, on=['subject_id', 'lifelog_date'], how='left')

feature_cols = [c for c in train_full.columns
                if c not in ['subject_id','lifelog_date','sleep_date'] + TARGETS]

X_train = train_full[feature_cols].copy()
X_test  = test_full[feature_cols].copy()
print(f'Total features: {len(feature_cols)}')

lgb_params_base = {
    'objective':         'binary',
    'metric':            'binary_logloss',
    'boosting_type':     'gbdt',
    'num_leaves':        31,
    'learning_rate':     0.02,
    'feature_fraction':  0.7,
    'bagging_fraction':  0.7,
    'bagging_freq':      5,
    'min_child_samples': 20,
    'reg_alpha':         0.3,
    'reg_lambda':        2.0,
    'n_estimators':      2000,
    'verbose':           -1,
    'n_jobs':            -1,
}

SEEDS   = [42, 1234, 9999, 7, 314, 2025, 777, 555]
N_FOLDS = 5

oof_preds  = np.zeros((len(X_train), len(TARGETS)))
test_preds = np.zeros((len(X_test),  len(TARGETS)))

for ti, target in enumerate(TARGETS):
    y = train_full[target].values
    print(f'\n  === Target: {target} | pos_rate: {y.mean():.3f} ===')

    all_oof  = np.zeros(len(X_train))
    all_test = np.zeros(len(X_test))
    n_models = 0

    for seed in SEEDS:
        params = {**lgb_params_base, 'random_state': seed}
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        seed_oof  = np.zeros(len(X_train))
        seed_test = np.zeros(len(X_test))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
            )
            seed_oof[val_idx]  = model.predict_proba(X_val)[:, 1]
            seed_test          += model.predict_proba(X_test)[:, 1] / N_FOLDS

        seed_oof_loss = log_loss(y, seed_oof)
        print(f'    seed={seed}: OOF log_loss={seed_oof_loss:.4f}')

        all_oof  += seed_oof
        all_test += seed_test
        n_models += 1

    oof_preds[:, ti]  = all_oof  / n_models
    test_preds[:, ti] = all_test / n_models
    print(f'  Ensemble OOF [{target}]: {log_loss(y, oof_preds[:, ti]):.4f}')

oof_total = np.mean([log_loss(train_full[t].values, oof_preds[:, i]) for i, t in enumerate(TARGETS)])
print(f'\n{"="*55}')
print(f'  v9 Total OOF: {oof_total:.4f}  (기대 Public ~ 0.6003)')
print(f'{"="*55}')
for i, t in enumerate(TARGETS):
    l = log_loss(train_full[t].values, oof_preds[:, i])
    print(f'  {t}: {l:.4f}')

report_data = {
    'avg_oof': oof_total,
    'n_features': len(feature_cols),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'per_target_oof': {t: log_loss(train_full[t].values, oof_preds[:, i]) for i, t in enumerate(TARGETS)},
}

# OOF / test_preds 저장
pd.DataFrame(oof_preds, columns=[f'oof_{t}' for t in TARGETS]).to_csv(OOF_PATH, index=False)
pd.DataFrame(test_preds, columns=TARGETS).to_csv(TEST_PREDS_PATH, index=False)

# 제출 파일
submission = sub_df[['subject_id','sleep_date','lifelog_date']].copy()
for i, t in enumerate(TARGETS):
    submission[t] = test_preds[:, i].clip(0.02, 0.98)

submission.to_csv(OUTPUT_PATH, index=False)
print(f'제출 파일: {OUTPUT_PATH} ({len(submission)}행)')
submission.head()

write_report(report_data)

summary = {
    'exp_tag': EXP_TAG,
    'avg_oof': oof_total,
    'per_target_oof': report_data['per_target_oof'],
    'n_features': len(feature_cols),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'artifacts': {
        'submission': OUTPUT_PATH,
        'report': REPORT_PATH,
        'summary': SUMMARY_PATH,
        'oof': OOF_PATH,
        'test_preds': TEST_PREDS_PATH,
        'run_log': RUN_LOG_PATH,
    },
    'timestamp': datetime.datetime.now().isoformat(),
}
with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'[요약 저장] {SUMMARY_PATH}')
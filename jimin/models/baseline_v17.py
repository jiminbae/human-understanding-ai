# 정세진의 base_v4 기반 수정
# 변경사항:
#   v17: base_v4 GBM 앙상블 + LSTM 블렌딩
#        LSTM: seq_len=7, hidden=32, dropout=0.5, 타겟별 학습
#        최종 예측 = GBM 85% + LSTM 15%

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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# ------------------------------------------------------------------
# LSTM 모델 정의
# ------------------------------------------------------------------
LSTM_SEQ_LEN  = 7    # lookback 일수
LSTM_HIDDEN   = 32   # hidden size (소형, 과적합 방지)
LSTM_DROPOUT  = 0.5
LSTM_EPOCHS   = 150
LSTM_LR       = 1e-3
LSTM_WD       = 1e-4
LSTM_PATIENCE = 20   # early stopping
LSTM_WEIGHT   = 0.15 # GBM 0.85 + LSTM 0.15

class DailyLSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, LSTM_HIDDEN, num_layers=2,
                            batch_first=True, dropout=LSTM_DROPOUT)
        self.drop = nn.Dropout(LSTM_DROPOUT)
        self.fc   = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(self.drop(out[:, -1, :]))).squeeze(-1)


def build_sequences(feat_matrix, subject_ids, dates, targets_arr, target_idx, seq_len):
    """피험자별로 seq_len일 슬라이딩 윈도우 시퀀스를 생성."""
    X_seq, y_seq, idx_seq = [], [], []
    df_tmp = pd.DataFrame({'subject_id': subject_ids, 'date': dates})
    df_tmp['row_idx'] = np.arange(len(df_tmp))
    for sid, grp in df_tmp.groupby('subject_id'):
        grp = grp.sort_values('date')
        idxs = grp['row_idx'].values
        for i in range(seq_len, len(idxs)):
            seq_idxs = idxs[i - seq_len: i]
            X_seq.append(feat_matrix[seq_idxs])
            y_seq.append(targets_arr[idxs[i], target_idx])
            idx_seq.append(idxs[i])
    return (np.array(X_seq, dtype=np.float32),
            np.array(y_seq,  dtype=np.float32),
            np.array(idx_seq))


def train_lstm_target(X_tr_seq, y_tr_seq, X_val_seq, n_features, device):
    """단일 타겟 LSTM 학습 후 검증/전체 시퀀스 예측 반환."""
    model = DailyLSTM(n_features).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WD)
    loss_fn = nn.BCELoss()

    tr_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)

    best_loss, patience_cnt, best_state = float('inf'), 0, None
    for epoch in range(LSTM_EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            l = loss_fn(pred, yb)
            l.backward()
            opt.step()
            ep_loss += l.item() * len(xb)
        ep_loss /= len(tr_ds)
        if ep_loss < best_loss - 1e-5:
            best_loss = ep_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= LSTM_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_t = torch.tensor(X_val_seq).to(device)
        preds = model(val_t).cpu().numpy()
    return preds


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

USE_TRAIN_SUBJ_NORM = os.environ.get('V17_TRAIN_NORM', '0') == '1'
USE_RANK_BLEND      = os.environ.get('V17_RANK_BLEND', '0') == '1'
USE_FOLD_SAFE_TE    = os.environ.get('V17_FOLD_SAFE_TE', '1') == '1'
USE_CALIBRATION     = os.environ.get('V17_CALIBRATION', '0') == '1'
CALIBRATION_METHOD  = os.environ.get('V17_CALIB_METHOD', 'platt').strip().lower()
if CALIBRATION_METHOD not in {'platt', 'isotonic'}:
    raise ValueError("V17_CALIB_METHOD must be one of: platt, isotonic")
PSEUDO_PUBLIC_TAIL_FRAC = float(os.environ.get('V17_PSEUDO_TAIL_FRAC', '0.2'))
if not (0 < PSEUDO_PUBLIC_TAIL_FRAC < 1):
    raise ValueError("V17_PSEUDO_TAIL_FRAC must be in (0, 1)")

FORCE_CPU = os.environ.get('V17_FORCE_CPU', '0') == '1'
HAS_CUDA  = (not FORCE_CPU) and torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f'CUDA: {torch.cuda.get_device_name(0) if HAS_CUDA else "not available"}')

BASE_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE_DIR / 'ch2025_data_items'
TRAIN_PATH = DATA_DIR / 'ch2026_metrics_train.csv'
SUB_PATH   = DATA_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR  = OUTPUTS_DIR / 'submissions'
REPORT_DIR  = OUTPUTS_DIR / 'report'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
OOF_DIR     = OUTPUTS_DIR / 'oof'
LOG_DIR     = OUTPUTS_DIR / 'log'

OUTPUT_PATH     = OUTPUT_DIR  / 'submission_v17.csv'
REPORT_PATH     = REPORT_DIR  / 'report_v17.txt'
SUMMARY_PATH    = SUMMARY_DIR / 'summary_v17.json'
OOF_PATH        = OOF_DIR     / 'oof_v17.csv'
TEST_PREDS_PATH = REPORT_DIR  / 'test_preds_v17.csv'
RUN_LOG_PATH    = LOG_DIR     / 'run_v17.log'


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def ensure_dirs():
    for d in [OUTPUT_DIR, REPORT_DIR, SUMMARY_DIR, OOF_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def agg_stats(vals, prefix):
    if len(vals) == 0:
        return {f'{prefix}_{k}': np.nan
                for k in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']}
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


# ------------------------------------------------------------------
# 센서 피처 추출 함수들
# ------------------------------------------------------------------

def extract_activity(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        acts = grp['m_activity'].values
        h    = grp['timestamp'].dt.hour.values
        for a in [0, 3, 4, 7, 8]:
            row[f'act_{a}_ratio'] = (acts == a).mean()
        row['act_active_ratio'] = ((acts == 7) | (acts == 8) | (acts == 3)).mean()
        row['act_still_ratio']  = (acts == 0).mean()
        row['act_n_records']    = len(acts)
        for seg, mask in [('morn', (h >= 6) & (h < 12)),
                          ('aftn', (h >= 12) & (h < 18)),
                          ('eve',  (h >= 18) & (h < 22)),
                          ('night',(h >= 22) | (h < 6))]:
            s_acts = acts[mask]
            row[f'act_{seg}_active'] = ((s_acts == 7) | (s_acts == 8)).mean() if len(s_acts) > 0 else np.nan
            row[f'act_{seg}_still']  = (s_acts == 0).mean()                   if len(s_acts) > 0 else np.nan
        pre = acts[(h >= 22) & (h < 24)]
        row['act_presleep_active'] = ((pre == 7) | (pre == 8)).mean() if len(pre) > 0 else np.nan
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
        row['screen_unlock_cnt'] = ((sc[1:] > sc[:-1])).sum() if len(sc) > 1 else 0
        for seg, mask in [('night',    (h >= 22) | (h < 2)),
                          ('eve',      (h >= 20) & (h <= 23)),
                          ('presleep', (h >= 22) & (h < 24))]:
            s_sc = sc[mask]
            row[f'screen_{seg}_on']    = (s_sc > 0).sum()
            row[f'screen_{seg}_ratio'] = (s_sc > 0).mean() if len(s_sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_light(df_raw, col, prefix, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row  = {'subject_id': sid, 'lifelog_date': d}
        vals = grp[col].dropna().values
        for k, v in agg_stats(vals, f'{prefix}_all').items():
            row[k] = v
        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('eve', (18, 22)), ('morn', (6, 10)), ('night', (22, 24))]:
            sv = grp.loc[h.between(lo, hi - 1), col].dropna().values
            row[f'{prefix}_{seg}_mean'] = safe_mean(sv)
        row[f'{prefix}_dark_ratio']   = (vals < 10).mean()   if len(vals) > 0 else np.nan
        row[f'{prefix}_bright_ratio'] = (vals > 1000).mean() if len(vals) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_ac(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        ch  = grp['m_charging'].values
        h   = grp['timestamp'].dt.hour.values
        row['ac_charging_ratio'] = ch.mean()
        for seg, mask in [('eve',      (h >= 21) & (h <= 23)),
                          ('night',    (h >= 22) | (h < 4)),
                          ('presleep', (h >= 22) & (h < 24))]:
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
        row['gps_lat_std']      = np.nanstd(lats)    if len(lats) > 0 else np.nan
        row['gps_lon_std']      = np.nanstd(lons)    if len(lons) > 0 else np.nan
        if len(lats) > 1:
            dlat = np.diff(lats); dlon = np.diff(lons)
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
        total_time = late_time = eve_time = n_apps = 0
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
                        all_bssids.add(net.get('bssid', ''))
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
        row   = {'subject_id': sid, 'lifelog_date': d}
        addrs = set()
        for v in grp['m_ble']:
            if isinstance(v, list):
                for dev in v:
                    if isinstance(dev, dict):
                        addrs.add(dev.get('address', ''))
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
        music_s = speech_s = silence_s = []
        music_s, speech_s, silence_s = [], [], []
        for v in grp['m_ambience']:
            if isinstance(v, list):
                d_map = {item[0]: item[1] for item in v
                         if isinstance(item, list) and len(item) == 2}
                music_s.append(d_map.get('Music', 0))
                speech_s.append(d_map.get('Speech', 0))
                silence_s.append(d_map.get('Silence', 0))
        row['amb_music_mean']   = np.mean(music_s)   if music_s   else np.nan
        row['amb_speech_mean']  = np.mean(speech_s)  if speech_s  else np.nan
        row['amb_silence_mean'] = np.mean(silence_s) if silence_s else np.nan
        row['amb_n_records']    = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)


# ------------------------------------------------------------------
# 수면 시간대 피처 추출
# ------------------------------------------------------------------

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
        row['slp_hr_deep_ratio']  = (sleep_hrs < 55).mean()  if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_awake_ratio'] = (sleep_hrs > 75).mean()  if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_light_ratio'] = ((sleep_hrs >= 55) & (sleep_hrs <= 75)).mean() if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 1:
            diffs = np.diff(sleep_hrs)
            row['slp_hr_rmssd'] = float(np.sqrt(np.nanmean(diffs ** 2)))
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
        row  = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        row['slp_pedo_steps']    = morn['step'].sum()
        row['slp_pedo_active']   = (morn['step'] > 5).sum()
        row['slp_pedo_calories'] = morn['burned_calories'].sum()
        row['slp_pedo_n_records']= len(morn)
        mid = grp[grp['timestamp'].dt.hour.between(2, 4)]
        row['slp_pedo_mid_steps']= mid['step'].sum()
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_activity(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row  = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) == 0:
            row.update({'slp_act_still_ratio': np.nan,
                        'slp_act_active_ratio': np.nan,
                        'slp_act_n_records': 0})
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
        row  = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            sc = morn['m_screen_use'].values
            row['slp_screen_on']    = (sc > 0).sum()
            row['slp_screen_ratio'] = (sc > 0).mean()
        else:
            row['slp_screen_on'] = np.nan
            row['slp_screen_ratio'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_light(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row  = {'subject_id': sid, 'sleep_date': d}
        morn = grp[grp['timestamp'].dt.hour < 9]
        if len(morn) > 0:
            vals = morn['w_light'].dropna().values
            row['slp_wlight_mean']  = safe_mean(vals)
            row['slp_wlight_dark']  = (vals < 5).mean()   if len(vals) > 0 else np.nan
            row['slp_wlight_light'] = (vals > 100).mean() if len(vals) > 0 else np.nan
        else:
            row['slp_wlight_mean'] = np.nan
            row['slp_wlight_dark'] = np.nan
            row['slp_wlight_light'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


# ------------------------------------------------------------------
# Target Encoding (단일 타겟 + cross-target)
# ------------------------------------------------------------------

def rank_norm(a):
    s = pd.Series(a)
    return (s.rank(method='average').values - 1) / max(len(s) - 1, 1)


def _build_subject_history(history_df, target):
    h = history_df[['subject_id', 'lifelog_date', target]].copy()
    h = h.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    hist = {}
    for sid, grp in h.groupby('subject_id'):
        hist[sid] = {
            'dates':  grp['lifelog_date'].to_numpy(),
            'labels': grp[target].to_numpy(),
        }
    return hist


def _encode_from_history(history_map, query_df, windows):
    rows = []
    for sid, d in query_df[['subject_id', 'lifelog_date']].itertuples(index=False):
        if sid not in history_map:
            row = {'te_lag1': np.nan}
            for w in windows:
                row[f'te_enc{w}'] = np.nan
            rows.append(row)
            continue
        dates  = history_map[sid]['dates']
        labels = history_map[sid]['labels']
        k      = np.searchsorted(dates, d, side='left')
        past   = labels[:k]
        row    = {'te_lag1': past[-1] if len(past) > 0 else np.nan}
        for w in windows:
            row[f'te_enc{w}'] = np.nanmean(past[-w:]) if len(past) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=query_df.index)


def build_fold_safe_target_encoding(train_hist_df, tr_query_df, val_query_df,
                                    test_query_df, target, windows):
    history_map = _build_subject_history(train_hist_df, target)
    tr_te   = _encode_from_history(history_map, tr_query_df,  windows)
    val_te  = _encode_from_history(history_map, val_query_df, windows)
    test_te = _encode_from_history(history_map, test_query_df, windows)
    return tr_te, val_te, test_te


def calibrate_probs(y_true, oof_prob, test_prob):
    oof_prob  = np.clip(oof_prob,  1e-7, 1 - 1e-7)
    test_prob = np.clip(test_prob, 1e-7, 1 - 1e-7)
    if CALIBRATION_METHOD == 'isotonic':
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(oof_prob, y_true)
        cal_oof  = cal.transform(oof_prob)
        cal_test = cal.transform(test_prob)
    else:
        cal = LogisticRegression(solver='lbfgs', max_iter=1000)
        cal.fit(oof_prob.reshape(-1, 1), y_true)
        cal_oof  = cal.predict_proba(oof_prob.reshape(-1, 1))[:, 1]
        cal_test = cal.predict_proba(test_prob.reshape(-1, 1))[:, 1]
    return np.clip(cal_oof, 1e-7, 1 - 1e-7), np.clip(cal_test, 1e-7, 1 - 1e-7)


def build_pseudo_public_mask(df, tail_frac):
    mask = pd.Series(False, index=df.index)
    for _, grp in df.sort_values(['subject_id', 'lifelog_date']).groupby('subject_id'):
        n      = len(grp)
        tail_n = max(1, int(np.ceil(n * tail_frac)))
        mask.loc[grp.index[-tail_n:]] = True
    return mask.values


# ------------------------------------------------------------------
# 피처 테이블 빌드
# ------------------------------------------------------------------

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
        ('mActivity',    extract_activity, None,      None),
        ('wPedo',        extract_pedo,     None,      None),
        ('wHr',          extract_hr,       None,      None),   # v14: 낮 HR 피처 실제 추출
        ('mScreenStatus',extract_screen,   None,      None),
        ('mLight',       extract_light,    'm_light', 'mlight'),
        ('wLight',       extract_wlight,   None,      None),
        ('mACStatus',    extract_ac,       None,      None),
        ('mGps',         extract_gps,      None,      None),
        ('mUsageStats',  extract_usage,    None,      None),
        ('mWifi',        extract_wifi,     None,      None),
        ('mBle',         extract_ble,      None,      None),
        ('mAmbience',    extract_ambience, None,      None),
    ]:
        print(f'  {name}...')
        df = load_parquet(name)
        feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))
        del df; gc.collect()

    print('Extracting sleep-date features...')
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
        del df; gc.collect()

    sleep_feats = sleep_feat_dfs[0]
    for df in sleep_feat_dfs[1:]:
        sleep_feats = sleep_feats.merge(df, on=['subject_id', 'sleep_date'], how='outer')

    feat_all = feat_dfs[0]
    for df in feat_dfs[1:]:
        feat_all = feat_all.merge(df, on=['subject_id', 'lifelog_date'], how='outer')

    feat_all['dow']        = feat_all['lifelog_date'].dt.dayofweek
    feat_all['month']      = feat_all['lifelog_date'].dt.month
    feat_all['week']       = feat_all['lifelog_date'].dt.isocalendar().week.astype(int)
    feat_all['is_weekend'] = (feat_all['dow'] >= 5).astype(int)
    feat_all['subject_num']= feat_all['subject_id'].str.extract(r'(\d+)').astype(int)
    feat_all['dow_sin']    = np.sin(2 * np.pi * feat_all['dow']   / 7)
    feat_all['dow_cos']    = np.cos(2 * np.pi * feat_all['dow']   / 7)
    feat_all['month_sin']  = np.sin(2 * np.pi * feat_all['month'] / 12)
    feat_all['month_cos']  = np.cos(2 * np.pi * feat_all['month'] / 12)

    if 'screen_on_total' in feat_all.columns and 'pedo_total_steps' in feat_all.columns:
        feat_all['screen_per_step'] = feat_all['screen_on_total'] / (feat_all['pedo_total_steps'] + 1)
    if 'pedo_total_calories' in feat_all.columns and 'usage_total_time' in feat_all.columns:
        feat_all['cal_per_usage'] = feat_all['pedo_total_calories'] / (feat_all['usage_total_time'] + 1)

    feat_all = feat_all.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)

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
        feat_all[f'{col}_lag1']   = g.shift(1)
        feat_all[f'{col}_lag2']   = g.shift(2)
        feat_all[f'{col}_diff1']  = feat_all[col] - feat_all[f'{col}_lag1']
        feat_all[f'{col}_ewma3']  = g.transform(lambda x: x.shift(1).ewm(span=3,  adjust=False).mean())
        feat_all[f'{col}_ewma7']  = g.transform(lambda x: x.shift(1).ewm(span=7,  adjust=False).mean())
        feat_all[f'{col}_roll3']  = g.transform(lambda x: x.shift(1).rolling(3,  min_periods=1).mean())
        feat_all[f'{col}_roll7']  = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
        feat_all[f'{col}_roll14'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())

    train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
    train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    test_full  = sub_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(
        feat_all, on=['subject_id', 'lifelog_date'], how='left')
    test_full  = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')

    numeric_cols = feat_all.select_dtypes(include=[np.number]).columns.tolist()
    exclude_from_norm = {'subject_num', 'dow', 'month', 'week', 'is_weekend',
                         'dow_sin', 'dow_cos', 'month_sin', 'month_cos'}
    norm_cols = [c for c in numeric_cols
                 if c not in exclude_from_norm
                 and 'lag' not in c and 'roll' not in c
                 and 'ewma' not in c and 'diff' not in c]

    if not USE_TRAIN_SUBJ_NORM:
        for col in norm_cols:
            mu  = feat_all.groupby('subject_id')[col].transform('mean')
            sig = feat_all.groupby('subject_id')[col].transform('std').replace(0, np.nan)
            feat_all[f'{col}_subj_z'] = (feat_all[col] - mu) / sig
        train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
        train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
        test_full  = sub_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(
            feat_all, on=['subject_id', 'lifelog_date'], how='left')
        test_full  = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    else:
        for col in norm_cols:
            tmp      = train_full[['subject_id', col]].copy()
            subj_mu  = tmp.groupby('subject_id')[col].mean()
            subj_std = tmp.groupby('subject_id')[col].std().replace(0, np.nan)
            global_mu  = tmp[col].mean()
            global_std = tmp[col].std()
            train_mu  = train_full['subject_id'].map(subj_mu)
            train_sig = train_full['subject_id'].map(subj_std)
            test_mu   = test_full['subject_id'].map(subj_mu).fillna(global_mu)
            test_sig  = test_full['subject_id'].map(subj_std).fillna(global_std)
            train_full[f'{col}_subj_z'] = (train_full[col] - train_mu) / train_sig
            test_full[f'{col}_subj_z']  = (test_full[col]  - test_mu)  / test_sig

    feature_cols = [c for c in train_full.columns
                    if c not in ['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS]
    print(f'Total features: {len(feature_cols)}')
    return train_full, test_full, feature_cols


# ------------------------------------------------------------------
# 학습 및 예측
# ------------------------------------------------------------------

def train_and_predict(train_full, test_full, feature_cols):
    X_train_base = train_full[feature_cols].copy()
    X_test_base  = test_full[feature_cols].copy()

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
        'early_stopping_rounds': 100,
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

    w_lgb, w_xgb, w_cat = 0.4, 0.3, 0.3
    seeds      = [42, 1234, 9999, 7, 314, 2025, 777, 555]
    n_folds    = 5
    te_windows = [3, 7, 14, 21]

    oof_preds  = np.zeros((len(X_train_base), len(TARGETS)))
    test_preds = np.zeros((len(X_test_base),  len(TARGETS)))

    for ti, target in enumerate(TARGETS):
        y = train_full[target].values
        print(f'\n=== Target: {target} | pos_rate: {y.mean():.3f} ===')

        all_oof_prob  = np.zeros(len(X_train_base))
        all_oof_rank  = np.zeros(len(X_train_base))
        all_test_prob = np.zeros(len(X_test_base))
        all_test_rank = np.zeros(len(X_test_base))

        for seed in seeds:
            skf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            seed_oof = np.zeros(len(X_train_base))
            seed_test_prob = np.zeros(len(X_test_base))

            for tr_idx, val_idx in skf.split(X_train_base, y):
                X_tr  = X_train_base.iloc[tr_idx].copy()
                X_val = X_train_base.iloc[val_idx].copy()
                X_te  = X_test_base.copy()
                y_tr, y_val = y[tr_idx], y[val_idx]

                if USE_FOLD_SAFE_TE:
                    hist_df   = train_full.iloc[tr_idx][['subject_id', 'lifelog_date', target]].copy()
                    tr_query  = train_full.iloc[tr_idx][['subject_id', 'lifelog_date']].copy()
                    val_query = train_full.iloc[val_idx][['subject_id', 'lifelog_date']].copy()
                    test_query= test_full[['subject_id', 'lifelog_date']].copy()

                    tr_te, val_te, test_te = build_fold_safe_target_encoding(
                        hist_df, tr_query, val_query, test_query, target, te_windows)

                    X_tr  = pd.concat([X_tr.reset_index(drop=True),
                                       tr_te.reset_index(drop=True)], axis=1)
                    X_val = pd.concat([X_val.reset_index(drop=True),
                                       val_te.reset_index(drop=True)], axis=1)
                    X_te  = pd.concat([X_te.reset_index(drop=True),
                                       test_te.reset_index(drop=True)], axis=1)

                model_lgb = lgb.LGBMClassifier(**{**lgb_params_base, 'random_state': seed})
                model_xgb = xgb.XGBClassifier(**{**xgb_params_base, 'random_state': seed})
                model_cat = CatBoostClassifier(**{**cat_params_base, 'random_seed': seed})

                try:
                    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                                  callbacks=[lgb.early_stopping(100, verbose=False),
                                             lgb.log_evaluation(-1)])
                except Exception:
                    cpu_params = dict(lgb_params_base); cpu_params['device'] = 'cpu'
                    model_lgb = lgb.LGBMClassifier(**{**cpu_params, 'random_state': seed})
                    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                                  callbacks=[lgb.early_stopping(100, verbose=False)])

                model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                              early_stopping_rounds=100, verbose=False)

                val_pred = (model_lgb.predict_proba(X_val)[:, 1] * w_lgb +
                            model_xgb.predict_proba(X_val)[:, 1] * w_xgb +
                            model_cat.predict_proba(X_val)[:, 1] * w_cat)
                te_pred  = (model_lgb.predict_proba(X_te)[:, 1] * w_lgb +
                            model_xgb.predict_proba(X_te)[:, 1] * w_xgb +
                            model_cat.predict_proba(X_te)[:, 1] * w_cat)

                seed_oof[val_idx]   = val_pred
                seed_test_prob     += te_pred / n_folds

            print(f'  seed={seed}: OOF log_loss={log_loss(y, seed_oof):.4f}')
            all_oof_prob  += seed_oof
            all_oof_rank  += rank_norm(seed_oof)
            all_test_prob += seed_test_prob
            all_test_rank += rank_norm(seed_test_prob)

        mean_oof_prob  = all_oof_prob  / len(seeds)
        mean_test_prob = all_test_prob / len(seeds)

        if USE_RANK_BLEND:
            mean_oof_rank  = all_oof_rank  / len(seeds)
            mean_test_rank = all_test_rank / len(seeds)
            target_oof  = 0.7 * mean_oof_prob  + 0.3 * mean_oof_rank
            target_test = 0.7 * mean_test_prob + 0.3 * mean_test_rank
        else:
            target_oof  = mean_oof_prob
            target_test = mean_test_prob

        if USE_CALIBRATION:
            target_oof, target_test = calibrate_probs(y, target_oof, target_test)

        oof_preds[:, ti]  = target_oof
        test_preds[:, ti] = target_test
        print(f'  GBM OOF [{target}]: {log_loss(y, oof_preds[:, ti]):.4f}')

    # ------------------------------------------------------------------
    # LSTM 블렌딩
    # ------------------------------------------------------------------
    print('\n=== LSTM 학습 시작 ===')
    lstm_device = torch.device('cuda' if HAS_CUDA else 'cpu')

    # 피처 행렬 구성 (NaN → 0 처리)
    feat_matrix_tr = X_train_base.fillna(0).values.astype(np.float32)
    feat_matrix_te = X_test_base.fillna(0).values.astype(np.float32)
    n_features = feat_matrix_tr.shape[1]

    # 피처 정규화 (LSTM은 스케일에 민감)
    feat_mean = feat_matrix_tr.mean(axis=0)
    feat_std  = feat_matrix_tr.std(axis=0) + 1e-8
    feat_matrix_tr = (feat_matrix_tr - feat_mean) / feat_std
    feat_matrix_te = (feat_matrix_te - feat_mean) / feat_std

    tr_sids  = train_full['subject_id'].values
    tr_dates = train_full['lifelog_date'].values
    te_sids  = test_full['subject_id'].values
    te_dates = test_full['lifelog_date'].values

    targets_arr = train_full[TARGETS].values.astype(np.float32)

    lstm_oof   = np.zeros((len(feat_matrix_tr), len(TARGETS)))
    lstm_test  = np.zeros((len(feat_matrix_te), len(TARGETS)))

    for ti, target in enumerate(TARGETS):
        print(f'  LSTM target={target}')

        # 전체 train 시퀀스
        X_all_seq, y_all_seq, idx_all = build_sequences(
            feat_matrix_tr, tr_sids, tr_dates, targets_arr, ti, LSTM_SEQ_LEN)

        # 테스트 시퀀스: train 마지막 (seq_len-1)일 + test 순서
        combined_feat = np.vstack([feat_matrix_tr, feat_matrix_te])
        combined_sids = np.concatenate([tr_sids, te_sids])
        combined_dates = np.concatenate([tr_dates, te_dates])
        dummy_targets = np.zeros((len(combined_feat), len(TARGETS)), dtype=np.float32)
        dummy_targets[:len(feat_matrix_tr)] = targets_arr

        X_te_seq, _, idx_te = build_sequences(
            combined_feat, combined_sids, combined_dates, dummy_targets, ti, LSTM_SEQ_LEN)
        te_mask = idx_te >= len(feat_matrix_tr)
        X_te_seq_only = X_te_seq[te_mask]
        idx_te_only   = idx_te[te_mask] - len(feat_matrix_tr)

        # 여러 시드로 학습 후 평균
        lstm_seeds = [42, 1234, 9999]
        seed_oof_preds  = np.zeros(len(feat_matrix_tr))
        seed_oof_counts = np.zeros(len(feat_matrix_tr))
        seed_te_preds   = np.zeros(len(feat_matrix_te))

        for ls in lstm_seeds:
            torch.manual_seed(ls)
            np.random.seed(ls)

            # temporal split: 각 피험자의 마지막 20%를 val로
            val_date_mask = np.zeros(len(feat_matrix_tr), dtype=bool)
            for sid in np.unique(tr_sids):
                sid_mask = tr_sids == sid
                sid_dates_sorted = np.sort(tr_dates[sid_mask])
                cutoff = sid_dates_sorted[int(len(sid_dates_sorted) * 0.8)]
                val_date_mask[sid_mask & (tr_dates >= cutoff)] = True

            tr_seq_mask  = ~val_date_mask[idx_all]
            val_seq_mask =  val_date_mask[idx_all]

            if tr_seq_mask.sum() < 10:
                continue

            preds_val = train_lstm_target(
                X_all_seq[tr_seq_mask], y_all_seq[tr_seq_mask],
                X_all_seq[val_seq_mask], n_features, lstm_device)
            seed_oof_preds[idx_all[val_seq_mask]]  += preds_val
            seed_oof_counts[idx_all[val_seq_mask]] += 1

            # 전체 train으로 재학습 후 test 예측
            preds_te = train_lstm_target(
                X_all_seq, y_all_seq, X_te_seq_only, n_features, lstm_device)
            seed_te_preds[idx_te_only] += preds_te / len(lstm_seeds)

        # val 예측 평균 (시퀀스 미생성 구간은 GBM 예측으로 채움)
        valid_mask = seed_oof_counts > 0
        seed_oof_preds[valid_mask] /= seed_oof_counts[valid_mask]
        seed_oof_preds[~valid_mask] = oof_preds[~valid_mask, ti]  # fallback

        lstm_oof[:, ti]  = seed_oof_preds
        lstm_test[:, ti] = seed_te_preds
        print(f'    LSTM OOF [{target}]: {log_loss(targets_arr[:, ti], lstm_oof[:, ti]):.4f}')

    # GBM + LSTM 블렌딩
    print(f'\n=== 블렌딩: GBM {1-LSTM_WEIGHT:.0%} + LSTM {LSTM_WEIGHT:.0%} ===')
    blended_oof  = (1 - LSTM_WEIGHT) * oof_preds  + LSTM_WEIGHT * lstm_oof
    blended_test = (1 - LSTM_WEIGHT) * test_preds + LSTM_WEIGHT * lstm_test
    for ti, target in enumerate(TARGETS):
        y = train_full[target].values
        print(f'  Blended OOF [{target}]: {log_loss(y, blended_oof[:, ti]):.4f}')

    return blended_oof, blended_test


# ------------------------------------------------------------------
# 리포트 / 서머리 저장
# ------------------------------------------------------------------

def write_report(report_data):
    lines = [
        '=' * 80,
        'Baseline v17 run report',
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        'Base: base_v4 + LSTM blend (GBM 85% + LSTM 15%)',
        f"  Total OOF:         {report_data['avg_oof']:.4f}",
        f"  Pseudo-public OOF: {report_data['pseudo_public_oof']:.4f}",
        f"  Feature count:     {report_data['n_features']}",
        '',
        '[Per target OOF]',
    ]
    for t, v in report_data['per_target_oof'].items():
        lines.append(f'  {t}: {v:.4f}')
    text = '\n'.join(lines)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print('\n' + text)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ensure_dirs()
    _run_log = open(RUN_LOG_PATH, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, _run_log)
    sys.stderr = Tee(sys.__stderr__, _run_log)

    print('Starting v17 Training Pipeline...')
    print(f'USE_FOLD_SAFE_TE={USE_FOLD_SAFE_TE}, USE_TRAIN_SUBJ_NORM={USE_TRAIN_SUBJ_NORM}, '
          f'USE_RANK_BLEND={USE_RANK_BLEND}, USE_CALIBRATION={USE_CALIBRATION}')

    train_df = pd.read_csv(TRAIN_PATH)
    sub_df   = pd.read_csv(SUB_PATH)
    for df in [train_df, sub_df]:
        df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
        df['sleep_date']   = pd.to_datetime(df['sleep_date'])

    train_full, test_full, feature_cols = build_feature_table(train_df, sub_df)
    oof_preds, test_preds = train_and_predict(train_full, test_full, feature_cols)

    per_target = {t: log_loss(train_full[t].values, oof_preds[:, i])
                  for i, t in enumerate(TARGETS)}
    oof_total  = float(np.mean(list(per_target.values())))

    pseudo_mask = build_pseudo_public_mask(
        train_full[['subject_id', 'lifelog_date']], PSEUDO_PUBLIC_TAIL_FRAC)
    pseudo_per_target = {t: log_loss(train_full.loc[pseudo_mask, t].values,
                                      oof_preds[pseudo_mask, i])
                         for i, t in enumerate(TARGETS)}
    pseudo_oof_total = float(np.mean(list(pseudo_per_target.values())))

    print(f'\n{"=" * 55}')
    print(f'v17 Total OOF:         {oof_total:.4f}')
    print(f'v17 Pseudo-public OOF: {pseudo_oof_total:.4f}')
    print(f'{"=" * 55}')

    # OOF CSV 저장
    oof_df = train_full[['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS].copy()
    for i, t in enumerate(TARGETS):
        oof_df[f'pred_{t}'] = oof_preds[:, i]
    oof_df.to_csv(OOF_PATH, index=False)

    # Test preds CSV 저장
    test_preds_df = test_full[['subject_id', 'lifelog_date', 'sleep_date']].copy()
    for i, t in enumerate(TARGETS):
        test_preds_df[t] = test_preds[:, i]
    test_preds_df.to_csv(TEST_PREDS_PATH, index=False)

    # Submission 저장
    submission = sub_df[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    for i, t in enumerate(TARGETS):
        submission[t] = test_preds[:, i].clip(0.02, 0.98)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f'submission saved: {OUTPUT_PATH}')

    report_data = {
        'avg_oof':           oof_total,
        'pseudo_public_oof': pseudo_oof_total,
        'per_target_oof':    per_target,
        'pseudo_per_target': pseudo_per_target,
        'n_features':        len(feature_cols),
        'n_train':           len(train_full),
        'n_test':            len(test_full),
    }
    write_report(report_data)

    summary = {
        'exp_tag': 'v17_gbm_lstm_blend',
        'use_train_subj_norm': USE_TRAIN_SUBJ_NORM,
        'use_rank_blend':      USE_RANK_BLEND,
        'use_fold_safe_te':    USE_FOLD_SAFE_TE,
        'use_calibration':     USE_CALIBRATION,
        'avg_oof':             oof_total,
        'pseudo_public_oof':   pseudo_oof_total,
        'per_target_oof':      per_target,
        'n_features':          len(feature_cols),
        'n_train':             len(train_full),
        'n_test':              len(test_full),
        'artifacts': {
            'submission':  str(OUTPUT_PATH),
            'report':      str(REPORT_PATH),
            'summary':     str(SUMMARY_PATH),
            'oof':         str(OOF_PATH),
            'test_preds':  str(TEST_PREDS_PATH),
            'run_log':     str(RUN_LOG_PATH),
        },
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'summary saved: {SUMMARY_PATH}')

    _run_log.close()


if __name__ == '__main__':
    main()

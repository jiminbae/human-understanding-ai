"""
제 5회 ETRI 휴먼이해 인공지능 논문경진대회 - v7a (Feature Boost)

[v6 대비 주요 변경사항]
  1. Lag 피처 복원 및 강화 (1/2/3일 lag + 7일 rolling mean/std)
  2. 피처 간 상호작용(Interaction) 피처 추가 (HR × 활동량, 수면 × 야간활동)
  3. 개인 상대 피처 확장 (25 → 40 컬럼, lag/rolling도 포함)
  4. 하루 시간대 구간별 센서 피처 (아침/오후/저녁/야간)
  5. 모든 v5 신규 피처 플래그 기본 ON
  6. 수면 연속성 피처 (Fragmentation index)
"""

import os, gc, json, warnings, datetime, sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna
import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

DATA_DIR        = './ch2025_data_items'
TRAIN_PATH      = './ch2026_metrics_train.csv'
SUBMISSION_PATH = './ch2026_submission_sample.csv'

# v7a: 모든 플래그 ON (피처 최대화)
USE_SLEEP_REFINED     = True
USE_APP_CATEGORY      = True
USE_PERSONAL_RELATIVE = True
USE_HR_FREQ           = True
USE_LAG_ROLLING       = True   # ★ 신규: Lag+Rolling 피처
USE_INTERACTION       = True   # ★ 신규: 상호작용 피처
USE_TIMEBLOCK         = True   # ★ 신규: 시간대별 피처

EXP_TAG     = "_v7a_feature_boost"
OUTPUT_PATH = f'./submission{EXP_TAG}.csv'
REPORT_PATH = f'./report{EXP_TAG}.txt'

TARGET_COLS   = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
N_SPLITS      = 10
OPTUNA_TRIALS = 100
SEED_LIST     = [42, 123, 777, 2024, 31337]
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
LGB_DEVICE    = 'gpu'  if DEVICE == 'cuda' else 'cpu'
XGB_DEVICE    = 'cuda' if DEVICE == 'cuda' else 'cpu'
CAT_DEVICE    = 'GPU'  if DEVICE == 'cuda' else 'CPU'

print(f"[Device] {DEVICE}")
if DEVICE == 'cuda':
    print(f"         {torch.cuda.get_device_name(0)}")
print(f"[Experiment] {EXP_TAG}")


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def to_list(x):
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    try:
        import ast
        return list(ast.literal_eval(str(x)))
    except:
        return []

def safe_mean(lst): return np.mean(lst) if len(lst) > 0 else np.nan
def safe_std(lst):  return np.std(lst)  if len(lst) > 0 else np.nan
def safe_min(lst):  return np.min(lst)  if len(lst) > 0 else np.nan
def safe_max(lst):  return np.max(lst)  if len(lst) > 0 else np.nan


# ──────────────────────────────────────────────
# 1. 기본 센서 피처 (v6 동일)
# ──────────────────────────────────────────────

def process_whr():
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_wHr.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    df['hour'] = df['timestamp'].dt.hour
    rows = []
    for (subj, date), grp in df.groupby(['subject_id', 'date']):
        all_hr, night_hr = [], []
        for _, row in grp.iterrows():
            lst = to_list(row['heart_rate'])
            all_hr.extend(lst)
            if row['hour'] < 7 or row['hour'] >= 22:
                night_hr.extend(lst)
        rows.append({
            'subject_id': subj, 'date': date,
            'hr_mean': safe_mean(all_hr), 'hr_std': safe_std(all_hr),
            'hr_min': safe_min(all_hr),   'hr_max': safe_max(all_hr),
            'hr_night_mean': safe_mean(night_hr),
            'hr_rmssd': (np.sqrt(np.mean(np.diff(all_hr)**2))
                         if len(all_hr) > 1 else np.nan),
        })
    del df; gc.collect()
    return pd.DataFrame(rows)


def process_gps():
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_mGps.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    rows = []
    for (subj, date), grp in df.groupby(['subject_id', 'date']):
        speeds = []
        for val in grp['m_gps']:
            for p in to_list(val):
                if isinstance(p, dict) and 'speed' in p:
                    speeds.append(float(p['speed']))
        rows.append({'subject_id': subj, 'date': date,
                     'gps_speed_mean': safe_mean(speeds),
                     'gps_speed_max':  safe_max(speeds),
                     'gps_speed_std':  safe_std(speeds)})
    del df; gc.collect()
    return pd.DataFrame(rows)


def process_usage():
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_mUsageStats.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    rows = []
    for (subj, date), grp in df.groupby(['subject_id', 'date']):
        total_time, n_apps = 0, 0
        for val in grp['m_usage_stats']:
            for p in to_list(val):
                if isinstance(p, dict):
                    total_time += p.get('total_time', 0)
                    n_apps += 1
        rows.append({'subject_id': subj, 'date': date,
                     'usage_total_time': total_time,
                     'usage_n_apps':     n_apps})
    del df; gc.collect()
    return pd.DataFrame(rows)


def process_ambience():
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_mAmbience.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    rows = []
    for (subj, date), grp in df.groupby(['subject_id', 'date']):
        scores, speech_cnt, total = [], 0, 0
        for val in grp['m_ambience']:
            lst = to_list(val)
            total += 1
            if lst and len(lst[0]) > 1:
                try: scores.append(float(lst[0][1]))
                except: pass
            if any('Speech' in str(i) for i in lst):
                speech_cnt += 1
        rows.append({'subject_id': subj, 'date': date,
                     'amb_top_score_mean': safe_mean(scores),
                     'amb_speech_ratio':   speech_cnt / max(total, 1)})
    del df; gc.collect()
    return pd.DataFrame(rows)


def process_wifi():
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_mWifi.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    rows = []
    for (subj, date), grp in df.groupby(['subject_id', 'date']):
        n_aps, rssis = [], []
        for val in grp['m_wifi']:
            lst = to_list(val)
            n_aps.append(len(lst))
            for p in lst:
                if isinstance(p, dict) and 'rssi' in p:
                    rssis.append(float(p['rssi']))
        rows.append({'subject_id': subj, 'date': date,
                     'wifi_n_ap_mean':  safe_mean(n_aps),
                     'wifi_rssi_mean':  safe_mean(rssis)})
    del df; gc.collect()
    return pd.DataFrame(rows)


def agg_numeric(df, value_cols, prefix):
    df = df.copy()
    df['date'] = df['timestamp'].dt.normalize()
    num_cols = [c for c in value_cols
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols: return None
    agg = df.groupby(['subject_id','date'])[num_cols].agg(['mean','std','min','max'])
    agg.columns = [f'{prefix}_{c}_{s}' for c, s in agg.columns]
    return agg.reset_index()


def process_sleep_timing():
    print("  sleep_timing ...")
    screen = pd.read_parquet(f'{DATA_DIR}/ch2025_mScreenStatus.parquet')
    screen = screen.sort_values(['subject_id','timestamp'])
    screen['date'] = screen['timestamp'].dt.normalize()
    rows = []
    for (subj, date), grp in screen.groupby(['subject_id','date']):
        grp = grp.sort_values('timestamp').reset_index(drop=True)
        ts, sc = grp['timestamp'].values, grp['m_screen_use'].values
        best_start, best_end, best_len = None, None, 0
        seg_start = None
        for t, s in zip(ts, sc):
            if s == 0:
                if seg_start is None: seg_start = t
            else:
                if seg_start is not None:
                    seg_len = (t - seg_start) / np.timedelta64(1, 'm')
                    if seg_len > best_len:
                        best_len, best_start, best_end = seg_len, seg_start, t
                    seg_start = None
        if seg_start is not None:
            seg_len = (ts[-1] - seg_start) / np.timedelta64(1, 'm')
            if seg_len > best_len:
                best_len, best_start, best_end = seg_len, seg_start, ts[-1]

        if best_len >= 30:
            sh = pd.Timestamp(best_start).hour + pd.Timestamp(best_start).minute/60
            eh = pd.Timestamp(best_end).hour   + pd.Timestamp(best_end).minute/60
        else:
            sh = eh = np.nan

        # ★ 수면 단편화 지수 추가: screen off 전환 횟수
        off_on_transitions = sum(1 for i in range(1, len(sc)) if sc[i-1]==0 and sc[i]==1)
        rows.append({'subject_id': subj, 'date': date,
                     'sleep_duration_min': best_len,
                     'sleep_start_hour':   sh,
                     'sleep_end_hour':     eh,
                     'screen_off_ratio':   (sc == 0).mean(),
                     'sleep_fragmentation': off_on_transitions})  # ★ 신규

    del screen; gc.collect()
    result = pd.DataFrame(rows)
    subj_std = result.groupby('subject_id')['sleep_start_hour'].std().reset_index()
    subj_std.columns = ['subject_id', 'sleep_regularity']
    result = result.merge(subj_std, on='subject_id', how='left')
    return result


def process_sleep_hr():
    print("  sleep_hr (야간 HRV) ...")
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_wHr.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    df['hour'] = df['timestamp'].dt.hour
    df_night = df[(df['hour'] < 7) | (df['hour'] >= 22)].copy()
    rows = []
    for (subj, date), grp in df_night.groupby(['subject_id','date']):
        all_hr = []
        for _, row in grp.iterrows():
            all_hr.extend(to_list(row['heart_rate']))
        if len(all_hr) > 1:
            diffs  = np.diff(all_hr)
            rmssd  = np.sqrt(np.mean(diffs**2))
            sdnn   = np.std(all_hr)
            pnn50  = np.mean(np.abs(diffs) > 50)
        else:
            rmssd = sdnn = pnn50 = np.nan
        rows.append({'subject_id': subj, 'date': date,
                     'night_hr_mean':  safe_mean(all_hr),
                     'night_hr_min':   safe_min(all_hr),
                     'night_rmssd':    rmssd,
                     'night_sdnn':     sdnn,
                     'night_pnn50':    pnn50,
                     'night_hr_range': safe_max(all_hr) - safe_min(all_hr)
                                       if len(all_hr) > 0 else np.nan})
    del df, df_night; gc.collect()
    return pd.DataFrame(rows)


def process_sleep_light():
    print("  sleep_light ...")
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_wLight.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    df['hour'] = df['timestamp'].dt.hour
    rows = []
    for (subj, date), grp in df.groupby(['subject_id','date']):
        night     = grp[(grp['hour'] < 7) | (grp['hour'] >= 22)]['w_light'].tolist()
        pre_sleep = grp[(grp['hour'] >= 21) & (grp['hour'] < 24)]['w_light'].tolist()
        rows.append({'subject_id': subj, 'date': date,
                     'light_night_mean':    safe_mean(night),
                     'light_night_max':     safe_max(night),
                     'light_presleep_mean': safe_mean(pre_sleep)})
    del df; gc.collect()
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# [★ 신규] 시간대별 센서 피처 (아침/오후/저녁/야간)
# ══════════════════════════════════════════════

TIME_BLOCKS = {
    'morning':  (6,  12),
    'afternoon':(12, 18),
    'evening':  (18, 22),
    'night':    (22, 30),  # 30 = 22~7 다음날
}

def process_timeblock_hr():
    """시간대별 HR 통계"""
    print("  [v7a] timeblock_hr ...")
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_wHr.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    df['hour'] = df['timestamp'].dt.hour

    rows = []
    for (subj, date), grp in df.groupby(['subject_id','date']):
        record = {'subject_id': subj, 'date': date}
        for block, (h_start, h_end) in TIME_BLOCKS.items():
            if h_end <= 24:
                mask = (grp['hour'] >= h_start) & (grp['hour'] < h_end)
            else:  # night: 22~7
                mask = (grp['hour'] >= h_start) | (grp['hour'] < (h_end - 24))
            sub = grp[mask]
            hrs = []
            for _, row in sub.iterrows():
                hrs.extend(to_list(row['heart_rate']))
            record[f'hr_{block}_mean'] = safe_mean(hrs)
            record[f'hr_{block}_std']  = safe_std(hrs)
        rows.append(record)
    del df; gc.collect()
    return pd.DataFrame(rows)


def process_timeblock_activity():
    """시간대별 활동량 (pedometer 기반)"""
    print("  [v7a] timeblock_activity ...")
    fpath = f'{DATA_DIR}/ch2025_wPedo.parquet'
    if not os.path.exists(fpath):
        return None
    df = pd.read_parquet(fpath)
    df['date'] = df['timestamp'].dt.normalize()
    df['hour'] = df['timestamp'].dt.hour
    rows = []
    for (subj, date), grp in df.groupby(['subject_id','date']):
        record = {'subject_id': subj, 'date': date}
        for block, (h_start, h_end) in TIME_BLOCKS.items():
            if h_end <= 24:
                mask = (grp['hour'] >= h_start) & (grp['hour'] < h_end)
            else:
                mask = (grp['hour'] >= h_start) | (grp['hour'] < (h_end - 24))
            sub = grp[mask]
            if 'step' in sub.columns:
                record[f'step_{block}_sum']  = sub['step'].sum() if len(sub) > 0 else 0
                record[f'step_{block}_mean'] = sub['step'].mean() if len(sub) > 0 else 0
        rows.append(record)
    del df; gc.collect()
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# v5 피처들 (그대로 유지)
# ══════════════════════════════════════════════

def process_sleep_refined():
    print("  [v5-①] sleep_refined ...")
    screen   = pd.read_parquet(f'{DATA_DIR}/ch2025_mScreenStatus.parquet')
    activity = pd.read_parquet(f'{DATA_DIR}/ch2025_mActivity.parquet')
    hr       = pd.read_parquet(f'{DATA_DIR}/ch2025_wHr.parquet')

    screen['minute']   = screen['timestamp'].dt.floor('min')
    activity['minute'] = activity['timestamp'].dt.floor('min')
    hr['minute']       = hr['timestamp'].dt.floor('min')

    hr['hr_mean_min'] = hr['heart_rate'].apply(
        lambda x: np.mean(to_list(x)) if len(to_list(x)) > 0 else np.nan)
    hr_min = hr[['subject_id','minute','hr_mean_min']].drop_duplicates(
        subset=['subject_id','minute'])

    merged = screen[['subject_id','minute','m_screen_use']].merge(
        activity[['subject_id','minute','m_activity']],
        on=['subject_id','minute'], how='outer')
    merged = merged.merge(hr_min, on=['subject_id','minute'], how='left')
    merged = merged.sort_values(['subject_id','minute']).reset_index(drop=True)

    subj_hr_means = merged.groupby('subject_id')['hr_mean_min'].mean().to_dict()
    merged['subj_hr_mean'] = merged['subject_id'].map(subj_hr_means)
    merged['screen_off']   = (merged['m_screen_use'].fillna(1) == 0).astype(int)
    merged['low_act']      = merged['m_activity'].isin([0, 3]).astype(int)
    merged['low_hr']       = (merged['hr_mean_min'] < merged['subj_hr_mean'] * 0.9).astype(int)
    merged['is_sleep']     = (merged['screen_off'] == 1) & \
                              ((merged['low_act'] == 1) | (merged['low_hr'] == 1))
    merged['is_sleep']     = merged['is_sleep'].astype(int)
    merged['date']         = merged['minute'].dt.normalize()

    rows = []
    for (subj, date), grp in merged.groupby(['subject_id','date']):
        grp = grp.sort_values('minute').reset_index(drop=True)
        sleep_mask = grp['is_sleep'].values
        best_len, best_start, best_end = 0, None, None
        seg_start = None
        for i, s in enumerate(sleep_mask):
            if s == 1:
                if seg_start is None: seg_start = i
            else:
                if seg_start is not None:
                    if (i - seg_start) > best_len:
                        best_len, best_start, best_end = i - seg_start, seg_start, i
                    seg_start = None
        if seg_start is not None and (len(sleep_mask) - seg_start) > best_len:
            best_len = len(sleep_mask) - seg_start
            best_start, best_end = seg_start, len(sleep_mask)

        if best_len >= 30 and best_start is not None:
            seg = grp.iloc[best_start:best_end]
            hr_in_sleep = seg['hr_mean_min'].dropna().tolist()
            refined_start_ts = seg['minute'].iloc[0]
            refined_end_ts   = seg['minute'].iloc[-1]
            # ★ 수면 중 HR 하락 속도 (처음 15분 vs 마지막 15분)
            hr_early = seg['hr_mean_min'].iloc[:15].dropna().tolist()
            hr_late  = seg['hr_mean_min'].iloc[-15:].dropna().tolist()
            hr_drop  = safe_mean(hr_early) - safe_mean(hr_late) if hr_early and hr_late else np.nan
            rows.append({
                'subject_id':             subj,
                'date':                   date,
                'refined_sleep_duration': best_len,
                'refined_sleep_start':    refined_start_ts.hour + refined_start_ts.minute/60,
                'refined_sleep_end':      refined_end_ts.hour + refined_end_ts.minute/60,
                'refined_sleep_hr_mean':  safe_mean(hr_in_sleep),
                'refined_sleep_hr_std':   safe_std(hr_in_sleep),
                'refined_sleep_hr_min':   safe_min(hr_in_sleep),
                'refined_sleep_ratio':    sleep_mask.mean(),
                'refined_sleep_hr_drop':  hr_drop,   # ★ 신규
            })
        else:
            rows.append({
                'subject_id':             subj, 'date': date,
                'refined_sleep_duration': 0,
                'refined_sleep_start':    np.nan, 'refined_sleep_end': np.nan,
                'refined_sleep_hr_mean':  np.nan, 'refined_sleep_hr_std': np.nan,
                'refined_sleep_hr_min':   np.nan, 'refined_sleep_ratio': np.nan,
                'refined_sleep_hr_drop':  np.nan,
            })

    del merged, screen, activity, hr, hr_min; gc.collect()
    return pd.DataFrame(rows)


APP_CATEGORIES = {
    'sns':          ['카카오톡','KakaoTalk','인스타','Instagram','페이스북','Facebook',
                     '트위터','Twitter','틱톡','TikTok','스냅','Snapchat','라인','Line',
                     '텔레그램','Telegram','디스코드','Discord','밴드','BAND','X (Twitter)'],
    'game':         ['게임','Game','포켓몬','Pokemon','리그','League','배그','PUBG',
                     '로블록스','Roblox','마인크래프트','Minecraft','쿠키','Cookie','Free Fire'],
    'productivity': ['메일','Mail','Gmail','캘린더','Calendar','노트','Note','문서','Docs',
                     '워드','Word','엑셀','Excel','PDF','드라이브','Drive','에버노트','Evernote',
                     '한글','Hangul','Slack','Zoom','Teams','뱅킹','Bank'],
    'media':        ['유튜브','YouTube','넷플릭스','Netflix','멜론','Melon','스포티파이','Spotify',
                     '왓챠','Watcha','웨이브','wavve','티빙','TVING','디즈니','Disney','쿠팡플레이',
                     'Apple Music','Video'],
    'shopping':     ['쿠팡','Coupang','네이버쇼핑','11번가','지마켓','Gmarket','배민','배달','요기요',
                     '무신사','Musinsa','당근','Market'],
    'news':         ['뉴스','News','네이버','NAVER','다음','Daum','구글','Google'],
}

def classify_app(app_name):
    if not app_name: return 'other'
    for cat, keywords in APP_CATEGORIES.items():
        for kw in keywords:
            if kw in app_name:
                return cat
    return 'other'

def process_app_category():
    print("  [v5-②] app_category ...")
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_mUsageStats.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    rows = []
    cat_keys = list(APP_CATEGORIES.keys()) + ['other']
    for (subj, date), grp in df.groupby(['subject_id', 'date']):
        cat_time = {k: 0 for k in cat_keys}
        for val in grp['m_usage_stats']:
            for p in to_list(val):
                if not isinstance(p, dict): continue
                app_name = p.get('app_name', '').strip()
                t = p.get('total_time', 0)
                cat = classify_app(app_name)
                cat_time[cat] += t
        record = {'subject_id': subj, 'date': date}
        total = sum(cat_time.values()) + 1e-8
        for k in cat_keys:
            record[f'app_{k}_time']  = cat_time[k]
            record[f'app_{k}_ratio'] = cat_time[k] / total
        rows.append(record)
    del df; gc.collect()
    return pd.DataFrame(rows)


def hr_frequency_features(hr_list):
    if len(hr_list) < 10:
        return dict(lf=np.nan, hf=np.nan, lf_hf_ratio=np.nan, spectral_entropy=np.nan)
    arr = np.array(hr_list, dtype=np.float64)
    arr = arr - np.mean(arr)
    n = len(arr)
    fft_vals = np.fft.rfft(arr)
    freqs    = np.fft.rfftfreq(n, d=1.0)
    power    = np.abs(fft_vals)**2
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    lf = float(np.sum(power[lf_mask]))
    hf = float(np.sum(power[hf_mask]))
    ratio = lf / (hf + 1e-8)
    power_norm = power / (np.sum(power) + 1e-8)
    entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
    return dict(lf=lf, hf=hf, lf_hf_ratio=ratio, spectral_entropy=entropy)

def process_hr_frequency():
    print("  [v5-④] hr_frequency (FFT) ...")
    df = pd.read_parquet(f'{DATA_DIR}/ch2025_wHr.parquet')
    df['date'] = df['timestamp'].dt.normalize()
    df['hour'] = df['timestamp'].dt.hour
    df_night = df[(df['hour'] < 7) | (df['hour'] >= 22)].copy()
    df_night = df_night.sort_values(['subject_id','timestamp'])
    rows = []
    for (subj, date), grp in df_night.groupby(['subject_id','date']):
        all_hr = []
        for _, row in grp.iterrows():
            all_hr.extend(to_list(row['heart_rate']))
        feat = hr_frequency_features(all_hr)
        rows.append({'subject_id': subj, 'date': date,
                     'hr_fft_lf':           feat['lf'],
                     'hr_fft_hf':           feat['hf'],
                     'hr_fft_lfhf_ratio':   feat['lf_hf_ratio'],
                     'hr_fft_spec_entropy': feat['spectral_entropy']})
    del df, df_night; gc.collect()
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 2. 피처 빌드
# ──────────────────────────────────────────────

def build_daily_features():
    print("=== 센서 피처 추출 ===")
    all_feat = None
    def merge_feat(base, new):
        if new is None or len(new) == 0: return base
        if base is None: return new
        return base.merge(new, on=['subject_id','date'], how='outer')

    for name, fn in [('wHr', process_whr), ('mGps', process_gps),
                     ('mUsage', process_usage), ('mAmbience', process_ambience),
                     ('mWifi', process_wifi)]:
        print(f"  {name} ...")
        try:
            feat = fn(); all_feat = merge_feat(all_feat, feat)
            del feat; gc.collect()
        except Exception as e:
            print(f"    [skip] {e}")

    simple_map = [
        ('wPedo',  'ch2025_wPedo.parquet',
         ['step','step_frequency','running_step','walking_step',
          'distance','speed','burned_calories']),
        ('wLight', 'ch2025_wLight.parquet',        ['w_light']),
        ('mLight', 'ch2025_mLight.parquet',        ['m_light']),
        ('mAct',   'ch2025_mActivity.parquet',     ['m_activity']),
        ('mScr',   'ch2025_mScreenStatus.parquet', ['m_screen_use']),
        ('mAC',    'ch2025_mACStatus.parquet',     ['m_charging']),
    ]
    for prefix, fname, cols in simple_map:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath): continue
        print(f"  {prefix} ...")
        df = pd.read_parquet(fpath)
        all_feat = merge_feat(all_feat, agg_numeric(df, cols, prefix))
        del df; gc.collect()

    print("  mScr(야간) ...")
    df_s = pd.read_parquet(f'{DATA_DIR}/ch2025_mScreenStatus.parquet')
    df_s['date'] = df_s['timestamp'].dt.normalize()
    df_s['hour'] = df_s['timestamp'].dt.hour
    night = df_s[(df_s['hour'] >= 22) | (df_s['hour'] < 7)]
    if len(night) > 0:
        ndf = night.groupby(['subject_id','date']).agg(
            screen_off_night_ratio=('m_screen_use', lambda x: (x==0).mean()),
            screen_night_count=('m_screen_use','count')).reset_index()
        all_feat = merge_feat(all_feat, ndf)
    del df_s, night; gc.collect()

    for fn in [process_sleep_timing, process_sleep_hr, process_sleep_light]:
        try:
            feat = fn(); all_feat = merge_feat(all_feat, feat)
            del feat; gc.collect()
        except Exception as e:
            print(f"    [skip] {e}")

    if USE_SLEEP_REFINED:
        try:
            feat = process_sleep_refined()
            all_feat = merge_feat(all_feat, feat)
            del feat; gc.collect()
        except Exception as e:
            print(f"    [skip sleep_refined] {e}")

    if USE_APP_CATEGORY:
        try:
            feat = process_app_category()
            all_feat = merge_feat(all_feat, feat)
            del feat; gc.collect()
        except Exception as e:
            print(f"    [skip app_category] {e}")

    if USE_HR_FREQ:
        try:
            feat = process_hr_frequency()
            all_feat = merge_feat(all_feat, feat)
            del feat; gc.collect()
        except Exception as e:
            print(f"    [skip hr_frequency] {e}")

    # ★ 신규: 시간대별 피처
    if USE_TIMEBLOCK:
        for fn_name, fn in [('timeblock_hr', process_timeblock_hr),
                             ('timeblock_activity', process_timeblock_activity)]:
            try:
                feat = fn()
                if feat is not None:
                    all_feat = merge_feat(all_feat, feat)
                del feat; gc.collect()
            except Exception as e:
                print(f"    [skip {fn_name}] {e}")

    print(f"  → 일별 피처 shape: {all_feat.shape}")
    return all_feat


# ──────────────────────────────────────────────
# 3. 날짜 / Lag+Rolling / 피험자 피처
# ──────────────────────────────────────────────

def add_date_features(df):
    for col in ['sleep_date', 'lifelog_date']:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_dow']        = df[col].dt.dayofweek
        df[f'{col}_month']      = df[col].dt.month
        df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5,6]).astype(int)
        df[f'{col}_days_since'] = (df[col] - pd.Timestamp("2024-01-01")).dt.days
    df['date_diff'] = (df['lifelog_date'] - df['sleep_date']).dt.days
    return df


def add_lag_features(train_df, test_df):
    """
    ★ v7a: 진짜 Lag 피처 구현
    핵심 센서 피처에 대해 1/2/3일 lag + 3/7일 rolling mean/std 추가
    """
    print("  [v7a] lag + rolling 피처 계산 중...")

    LAG_COLS = [
        'hr_mean', 'hr_std', 'hr_rmssd',
        'night_rmssd', 'night_sdnn',
        'sleep_duration_min', 'sleep_start_hour', 'screen_off_ratio',
        'wPedo_step_mean', 'wPedo_distance_mean',
        'usage_total_time', 'amb_speech_ratio',
    ]
    LAG_DAYS     = [1, 2, 3]
    ROLLING_DAYS = [3, 7]

    combined = pd.concat([train_df, test_df], axis=0, sort=False)
    combined = combined.sort_values(['subject_id', 'lifelog_date'])
    combined['lifelog_date'] = pd.to_datetime(combined['lifelog_date'])

    valid_lag_cols = [c for c in LAG_COLS if c in combined.columns]
    new_cols = []

    for col in valid_lag_cols:
        grp = combined.groupby('subject_id')[col]

        for lag in LAG_DAYS:
            new_col = f'{col}_lag{lag}'
            combined[new_col] = grp.shift(lag)
            new_cols.append(new_col)

        for win in ROLLING_DAYS:
            mean_col = f'{col}_roll{win}m'
            std_col  = f'{col}_roll{win}s'
            combined[mean_col] = grp.shift(1).transform(
                lambda x: x.rolling(win, min_periods=1).mean())
            combined[std_col]  = grp.shift(1).transform(
                lambda x: x.rolling(win, min_periods=2).std())
            new_cols.extend([mean_col, std_col])

    train_out = combined[combined.index.isin(train_df.index)].copy()
    test_out  = combined[combined.index.isin(test_df.index)].copy()

    print(f"    → lag/rolling 피처 {len(new_cols)}개 추가")
    return train_out, test_out


def add_interaction_features(train_df, test_df):
    """
    ★ v7a 신규: 피처 간 상호작용
    - HR × 활동량
    - 수면시간 × 야간 HR 변동
    - SNS 사용시간 × 수면 품질
    - 수면 규칙성 × 수면 시간
    """
    print("  [v7a] interaction 피처 추가 중...")
    for df in [train_df, test_df]:
        # HR × 활동 상호작용
        if 'hr_mean' in df.columns and 'wPedo_step_mean' in df.columns:
            df['interact_hr_x_step'] = df['hr_mean'].fillna(0) * df['wPedo_step_mean'].fillna(0)

        # 수면시간 × 야간 HRV
        if 'sleep_duration_min' in df.columns and 'night_rmssd' in df.columns:
            df['interact_sleep_x_hrv'] = df['sleep_duration_min'].fillna(0) * df['night_rmssd'].fillna(0)

        # 수면 시간의 표준화 대비 야간 HR
        if 'sleep_duration_min' in df.columns and 'night_hr_mean' in df.columns:
            df['interact_sleep_x_night_hr'] = (
                df['sleep_duration_min'].fillna(0) / (df['night_hr_mean'].fillna(60) + 1e-6))

        # LF/HF × 수면 단편화
        if 'hr_fft_lfhf_ratio' in df.columns and 'sleep_fragmentation' in df.columns:
            df['interact_lfhf_x_frag'] = (
                df['hr_fft_lfhf_ratio'].fillna(0) * df['sleep_fragmentation'].fillna(0))

        # SNS 시간 × 수면 시작 시간 (늦은 SNS → 늦은 취침)
        if 'app_sns_time' in df.columns and 'sleep_start_hour' in df.columns:
            df['interact_sns_x_sleep_start'] = (
                df['app_sns_time'].fillna(0) * df['sleep_start_hour'].fillna(0))

        # 수면 규칙성 × 수면 지속시간
        if 'sleep_regularity' in df.columns and 'sleep_duration_min' in df.columns:
            df['interact_regularity_x_duration'] = (
                df['sleep_regularity'].fillna(0) * df['sleep_duration_min'].fillna(0))

        # 저녁 HR vs 아침 HR (각성 정도)
        if 'hr_evening_mean' in df.columns and 'hr_morning_mean' in df.columns:
            df['interact_hr_evening_vs_morning'] = (
                df['hr_evening_mean'].fillna(0) - df['hr_morning_mean'].fillna(0))

    return train_df, test_df


def add_subject_stats(train_df, test_df, feat_cols):
    num_cols = [c for c in feat_cols
                if pd.api.types.is_numeric_dtype(train_df[c])][:30]
    stats = train_df.groupby('subject_id')[num_cols].agg(['mean','std']).reset_index()
    stats.columns = (['subject_id'] +
                     [f'subj_{c}_{s}' for c in num_cols for s in ['mean','std']])
    train_df = train_df.merge(stats, on='subject_id', how='left')
    test_df  = test_df.merge(stats,  on='subject_id', how='left')
    return train_df, test_df


def add_personal_relative_features(train_df, test_df, feat_cols):
    """★ v7a: 피처 한도 25 → 40으로 확장 + lag 피처도 포함"""
    print("  [v7a] personal_relative (확장) ...")
    candidate = [c for c in feat_cols if any(k in c for k in [
        'hr_', 'night_', 'sleep_', 'step', 'distance', 'speed',
        'usage_total', 'wPedo', 'light', 'refined_sleep',
        'lag', 'roll',  # ★ lag/rolling 피처도 포함
    ])]
    candidate = [c for c in candidate
                 if c in train_df.columns
                 and pd.api.types.is_numeric_dtype(train_df[c])]
    candidate = candidate[:40]  # ★ 40개로 확장

    subj_mean = train_df.groupby('subject_id')[candidate].mean()
    subj_std  = train_df.groupby('subject_id')[candidate].std()

    for df in [train_df, test_df]:
        for c in candidate:
            mean_map = subj_mean[c].to_dict()
            std_map  = subj_std[c].to_dict()
            df[f'{c}_rel_diff']   = df[c] - df['subject_id'].map(mean_map)
            df[f'{c}_rel_zscore'] = (df[c] - df['subject_id'].map(mean_map)) / \
                                    (df['subject_id'].map(std_map).replace(0, 1) + 1e-6)
    return train_df, test_df


# ──────────────────────────────────────────────
# 4. Calibration
# ──────────────────────────────────────────────

def calibrate(val_p, val_y, test_p):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_p, val_y)
    return (np.clip(ir.transform(test_p), 0.0, 1.0),
            np.clip(ir.transform(val_p),  0.0, 1.0))


# ──────────────────────────────────────────────
# 5. Optuna (v6 동일)
# ──────────────────────────────────────────────

def tune_lgb(X, y, n_trials=OPTUNA_TRIALS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_est', 300, 2000),
            learning_rate=trial.suggest_float('lr', 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int('leaves', 15, 127),
            min_child_samples=trial.suggest_int('min_child', 5, 50),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample', 0.5, 1.0),
            reg_alpha=trial.suggest_float('alpha', 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float('lambda', 1e-3, 10, log=True),
            class_weight='balanced', device=LGB_DEVICE, random_state=SEED,
            verbose=-1, force_col_wise=True,
        )
        losses = []
        for tr_i, val_i in skf.split(X, y):
            m = LGBMClassifier(**params)
            m.fit(X.iloc[tr_i], y.iloc[tr_i],
                  eval_set=[(X.iloc[val_i], y.iloc[val_i])],
                  callbacks=[early_stopping(30), log_evaluation(-1)])
            p = np.clip(m.predict_proba(X.iloc[val_i])[:,1], 1e-7, 1-1e-7)
            losses.append(log_loss(y.iloc[val_i], p))
        return np.mean(losses)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_xgb(X, y, n_trials=OPTUNA_TRIALS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_f = X.fillna(-999)
    pw  = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_est', 300, 2000),
            learning_rate=trial.suggest_float('lr', 0.005, 0.1, log=True),
            max_depth=trial.suggest_int('depth', 3, 10),
            min_child_weight=trial.suggest_int('min_child', 1, 20),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample', 0.5, 1.0),
            reg_alpha=trial.suggest_float('alpha', 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float('lambda', 1e-3, 10, log=True),
            scale_pos_weight=pw, device=XGB_DEVICE, eval_metric='logloss',
            early_stopping_rounds=30, random_state=SEED, verbosity=0,
        )
        losses = []
        for tr_i, val_i in skf.split(X_f, y):
            m = XGBClassifier(**params)
            m.fit(X_f.iloc[tr_i], y.iloc[tr_i],
                  eval_set=[(X_f.iloc[val_i], y.iloc[val_i])], verbose=False)
            p = np.clip(m.predict_proba(X_f.iloc[val_i])[:,1], 1e-7, 1-1e-7)
            losses.append(log_loss(y.iloc[val_i], p))
        return np.mean(losses)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_cat(X, y, cat_idx, n_trials=OPTUNA_TRIALS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pw  = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    def objective(trial):
        params = dict(
            iterations=trial.suggest_int('iter', 300, 2000),
            learning_rate=trial.suggest_float('lr', 0.005, 0.1, log=True),
            depth=trial.suggest_int('depth', 4, 10),
            l2_leaf_reg=trial.suggest_float('l2', 1e-3, 10, log=True),
            bagging_temperature=trial.suggest_float('bagging_temp', 0, 1),
            random_strength=trial.suggest_float('rand_str', 0, 3),
            class_weights=[1.0, pw], task_type=CAT_DEVICE,
            early_stopping_rounds=30, random_seed=SEED, verbose=0,
        )
        losses = []
        for tr_i, val_i in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[tr_i], y.iloc[tr_i], cat_features=cat_idx,
                  eval_set=(X.iloc[val_i], y.iloc[val_i]))
            p = np.clip(m.predict_proba(X.iloc[val_i])[:,1], 1e-7, 1-1e-7)
            losses.append(log_loss(y.iloc[val_i], p))
        return np.mean(losses)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ──────────────────────────────────────────────
# 6. Seed Ensemble 학습 (v6 동일)
# ──────────────────────────────────────────────

def train_predict_seed(X, y, X_test, name, lgb_p, xgb_p, cat_p, seed):
    lgb_params = {
        'n_estimators':lgb_p['n_est'], 'learning_rate':lgb_p['lr'],
        'num_leaves':lgb_p['leaves'], 'min_child_samples':lgb_p['min_child'],
        'subsample':lgb_p['subsample'], 'colsample_bytree':lgb_p['colsample'],
        'reg_alpha':lgb_p['alpha'], 'reg_lambda':lgb_p['lambda'],
    }
    xgb_params = {
        'n_estimators':xgb_p['n_est'], 'learning_rate':xgb_p['lr'],
        'max_depth':xgb_p['depth'], 'min_child_weight':xgb_p['min_child'],
        'subsample':xgb_p['subsample'], 'colsample_bytree':xgb_p['colsample'],
        'reg_alpha':xgb_p['alpha'], 'reg_lambda':xgb_p['lambda'],
    }
    cat_params = {
        'iterations':cat_p['iter'], 'learning_rate':cat_p['lr'],
        'depth':cat_p['depth'], 'l2_leaf_reg':cat_p['l2'],
        'bagging_temperature':cat_p['bagging_temp'],
        'random_strength':cat_p['rand_str'],
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    test_pred = np.zeros(len(X_test))
    oof_pred  = np.zeros(len(X))
    cat_idx   = [i for i, c in enumerate(X.columns)
                 if X[c].dtype == object or str(X[c].dtype) == 'category']
    pw = float((y == 0).sum()) / max(float((y == 1).sum()), 1)

    for fold, (tr_i, val_i) in enumerate(skf.split(X, y)):
        X_tr, y_tr   = X.iloc[tr_i],  y.iloc[tr_i]
        X_val, y_val = X.iloc[val_i], y.iloc[val_i]

        lgb = LGBMClassifier(**lgb_params, device=LGB_DEVICE, random_state=seed,
                             class_weight='balanced', verbose=-1, force_col_wise=True)
        lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(50), log_evaluation(-1)])
        lgb_t, lgb_v = calibrate(lgb.predict_proba(X_val)[:,1], y_val,
                                  lgb.predict_proba(X_test)[:,1])

        Xtr_x, Xval_x, Xtest_x = (X_tr.fillna(-999), X_val.fillna(-999), X_test.fillna(-999))
        xgb = XGBClassifier(**xgb_params, device=XGB_DEVICE, scale_pos_weight=pw,
                            eval_metric='logloss', early_stopping_rounds=50,
                            random_state=seed, verbosity=0)
        xgb.fit(Xtr_x, y_tr, eval_set=[(Xval_x, y_val)], verbose=False)
        xgb_t, xgb_v = calibrate(xgb.predict_proba(Xval_x)[:,1], y_val,
                                   xgb.predict_proba(Xtest_x)[:,1])

        cat = CatBoostClassifier(**cat_params, task_type=CAT_DEVICE,
                                 class_weights=[1.0, pw], early_stopping_rounds=50,
                                 random_seed=seed, verbose=0)
        cat.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_val, y_val))
        cat_t, cat_v = calibrate(cat.predict_proba(X_val)[:,1], y_val,
                                  cat.predict_proba(X_test)[:,1])

        fold_t = np.clip(0.3*lgb_t + 0.3*xgb_t + 0.4*cat_t, 1e-7, 1-1e-7)
        fold_v = np.clip(0.3*lgb_v + 0.3*xgb_v + 0.4*cat_v, 1e-7, 1-1e-7)
        test_pred       += fold_t / N_SPLITS
        oof_pred[val_i]  = fold_v

    oof_pred = np.clip(oof_pred, 1e-7, 1-1e-7)
    return test_pred, oof_pred, log_loss(y, oof_pred)


def train_predict_ensemble(X, y, X_test, name, lgb_p, xgb_p, cat_p):
    test_preds, oof_preds, seed_losses = [], [], []
    for seed in SEED_LIST:
        test_p, oof_p, seed_l = train_predict_seed(
            X, y, X_test, name, lgb_p, xgb_p, cat_p, seed)
        test_preds.append(test_p); oof_preds.append(oof_p); seed_losses.append(seed_l)
        print(f"    seed={seed:5d} | {name} | OOF: {seed_l:.4f}")
    final_test = np.mean(test_preds, axis=0)
    final_oof  = np.mean(oof_preds, axis=0)
    final_loss = log_loss(y, np.clip(final_oof, 1e-7, 1-1e-7))
    print(f"  >> Seed-Ensemble OOF: {final_loss:.4f} (std={np.std(seed_losses):.4f})")
    return final_test, final_oof, final_loss, seed_losses


# ──────────────────────────────────────────────
# 7. 리포트
# ──────────────────────────────────────────────

def write_report(report_data, path=REPORT_PATH):
    lines = ["="*80,
             f"ETRI 휴먼이해 AI 논문경진대회 - 실험 리포트 ({EXP_TAG})",
             f"실행 시각: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "="*80, "",
             f"[실험 설정]",
             f"  Tag: {EXP_TAG} | Device: {DEVICE}",
             f"  Feature count: {report_data['n_features']} | Train: {report_data['n_train']} | Test: {report_data['n_test']}",
             f"  N_SPLITS: {N_SPLITS} | Optuna: {OPTUNA_TRIALS} | Seeds: {SEED_LIST}", "",
             "="*80, f"[전체 결과]  Average OOF Log-Loss: {report_data['avg_loss']:.4f}", "="*80,
             f"{'Target':<8}{'OOF':<10}{'Seed std':<12}", "-"*50]
    for t in TARGET_COLS:
        d = report_data['per_target'][t]
        lines.append(f"{t:<8}{d['oof']:<10.4f}{np.std(d['seed_losses']):<12.4f}")

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("\n" + text)
    print(f"\n[리포트 저장] {path}")


# ──────────────────────────────────────────────
# 8. 메인
# ──────────────────────────────────────────────

def main():
    print("\n[1] 데이터 로딩")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(SUBMISSION_PATH)

    print("[2] 날짜 피처")
    train_df = add_date_features(train_df)
    test_df  = add_date_features(test_df)

    print("[3] 센서 피처")
    daily = build_daily_features()
    daily['date'] = pd.to_datetime(daily['date'])
    train_df = train_df.merge(
        daily.rename(columns={'date':'lifelog_date'}),
        on=['subject_id','lifelog_date'], how='left')
    test_df = test_df.merge(
        daily.rename(columns={'date':'lifelog_date'}),
        on=['subject_id','lifelog_date'], how='left')
    del daily; gc.collect()

    print("[4] Lag + Rolling 피처")
    if USE_LAG_ROLLING:
        train_df, test_df = add_lag_features(train_df, test_df)

    drop = ['subject_id','sleep_date','lifelog_date'] + TARGET_COLS
    feat_cols = [c for c in train_df.columns if c not in drop]

    print("[5] 피험자 통계 피처")
    train_df, test_df = add_subject_stats(train_df, test_df, feat_cols)
    feat_cols = [c for c in train_df.columns if c not in drop]

    print("[5b] 개인 상대 피처")
    if USE_PERSONAL_RELATIVE:
        train_df, test_df = add_personal_relative_features(train_df, test_df, feat_cols)
        feat_cols = [c for c in train_df.columns if c not in drop]

    print("[5c] 상호작용 피처")
    if USE_INTERACTION:
        train_df, test_df = add_interaction_features(train_df, test_df)
        feat_cols = [c for c in train_df.columns if c not in drop]

    X      = train_df[feat_cols].fillna(-1)
    X_test = test_df[feat_cols].fillna(-1)
    cat_idx = [i for i, c in enumerate(X.columns)
               if X[c].dtype == object or str(X[c].dtype) == 'category']
    print(f"  피처: {X.shape[1]} | 학습: {len(X)} | 테스트: {len(X_test)}")

    print("\n[6] 모델 학습")
    sub = test_df[['subject_id','sleep_date','lifelog_date']].copy()
    sub['sleep_date']   = sub['sleep_date'].dt.strftime('%Y-%m-%d')
    sub['lifelog_date'] = sub['lifelog_date'].dt.strftime('%Y-%m-%d')

    report_data = {'n_features': X.shape[1], 'n_train': len(X),
                   'n_test': len(X_test), 'per_target': {}}
    losses = []
    for t in TARGET_COLS:
        print(f"\n  ▶ {t}")
        y = train_df[t]
        lgb_p, lgb_opt = tune_lgb(X, y)
        xgb_p, xgb_opt = tune_xgb(X, y)
        cat_p, cat_opt = tune_cat(X, y, cat_idx)
        print(f"    Optuna best — LGB: {lgb_opt:.4f} / XGB: {xgb_opt:.4f} / CAT: {cat_opt:.4f}")

        preds, oof, final_loss, seed_losses = train_predict_ensemble(
            X, y, X_test, t, lgb_p, xgb_p, cat_p)
        sub[t] = np.clip(preds, 1e-7, 1-1e-7)
        losses.append(final_loss)
        report_data['per_target'][t] = {
            'oof': final_loss, 'seed_losses': seed_losses,
            'lgb_params': lgb_p, 'xgb_params': xgb_p, 'cat_params': cat_p,
            'lgb_opt': lgb_opt, 'xgb_opt': xgb_opt, 'cat_opt': cat_opt,
            'test_preds': sub[t].values,
        }

    report_data['avg_loss'] = np.mean(losses)
    sub.to_csv(OUTPUT_PATH, index=False)
    write_report(report_data)
    print(f"\n[제출 파일] {OUTPUT_PATH}")

    summary = {
        'exp_tag': EXP_TAG, 'avg_oof': report_data['avg_loss'],
        'n_features': X.shape[1],
        'per_target_oof': {t: report_data['per_target'][t]['oof'] for t in TARGET_COLS},
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(f'./summary{EXP_TAG}.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
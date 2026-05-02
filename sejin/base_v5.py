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
from sklearn.isotonic import IsotonicRegression
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna 진행 상황 요약

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
# v13 Master: HR Features + Optuna Leakage Fix + LGBM Meta Stacking + Gain Select + TE Smoothing
USE_TRAIN_SUBJ_NORM = os.environ.get('V12_TRAIN_NORM', '0') == '1'
USE_RANK_BLEND = os.environ.get('V12_RANK_BLEND', '0') == '1'
USE_FOLD_SAFE_TE = os.environ.get('V12_FOLD_SAFE_TE', '1') == '1'
USE_CALIBRATION = os.environ.get('V12_CALIBRATION', '0') == '1'
CALIBRATION_METHOD = os.environ.get('V12_CALIB_METHOD', 'platt').strip().lower()

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

EXP_TAG = '_master_v13_optimized'
OUTPUT_PATH = OUTPUT_DIR / f'submission{EXP_TAG}.csv'
REPORT_PATH = REPORT_DIR / f'report{EXP_TAG}.txt'

# ---------------------------------------------------------------------
# 1. 유틸리티 함수
# ---------------------------------------------------------------------
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for stream in self.streams: stream.write(data); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()

def ensure_dirs():
    for d in [OUTPUT_DIR, REPORT_DIR, SUMMARY_DIR, OOF_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

def agg_stats(vals, prefix):
    if len(vals) == 0:
        return {f'{prefix}_mean': np.nan, f'{prefix}_std': np.nan, f'{prefix}_min': np.nan, f'{prefix}_max': np.nan, f'{prefix}_median': np.nan}
    return {f'{prefix}_mean': np.nanmean(vals), f'{prefix}_std': np.nanstd(vals), f'{prefix}_min': np.nanmin(vals), f'{prefix}_max': np.nanmax(vals), f'{prefix}_median': np.nanmedian(vals)}

def safe_mean(vals):
    arr = np.array(vals)
    return np.nanmean(arr) if len(arr) > 0 else np.nan

def load_parquet(name):
    df = pd.read_parquet(DATA_DIR / f'ch2025_{name}.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ---------------------------------------------------------------------
# 🔥 [1순위 반영] 심박수(HR) 피처 전면 재구현
# ---------------------------------------------------------------------
def extract_hr(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        
        # 내부 헬퍼: HR 배열 평탄화 및 추출
        def get_hr_array(series):
            vals = []
            for v in series:
                if isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v, dtype=float).ravel()
                    vals.extend(arr[arr > 0].tolist())
                elif isinstance(v, (int, float)) and v > 0:
                    vals.append(v)
            return np.array(vals)

        daily_hr = get_hr_array(grp['heart_rate'])
        if len(daily_hr) > 0:
            row['hr_daily_mean'] = np.nanmean(daily_hr)
            row['hr_daily_std'] = np.nanstd(daily_hr)
            # 심박 변동성 (RMSSD)
            row['hr_daily_rmssd'] = float(np.sqrt(np.nanmean(np.diff(daily_hr)**2))) if len(daily_hr) > 1 else np.nan
        else:
            row['hr_daily_mean'], row['hr_daily_std'], row['hr_daily_rmssd'] = np.nan, np.nan, np.nan

        # 시간대별 심박수 (아침, 오후, 저녁, 야간)
        h = grp['timestamp'].dt.hour
        for seg, (lo, hi) in [('morn', (6, 12)), ('aftn', (12, 18)), ('eve', (18, 22)), ('night', (22, 24))]:
            seg_hr = get_hr_array(grp.loc[h.between(lo, hi - 1), 'heart_rate'])
            row[f'hr_{seg}_mean'] = np.nanmean(seg_hr) if len(seg_hr) > 0 else np.nan
            row[f'hr_{seg}_std'] = np.nanstd(seg_hr) if len(seg_hr) > 0 else np.nan
            
        feats.append(row)
        
    df_feats = pd.DataFrame(feats)
    
    # 개인 기준 상대적 HR (오늘 HR - 해당 피험자 전체 평균 HR)
    if 'hr_daily_mean' in df_feats.columns:
        subj_mean = df_feats.groupby('subject_id')['hr_daily_mean'].transform('mean')
        df_feats['hr_daily_rel_mean'] = df_feats['hr_daily_mean'] - subj_mean
        
    return keys.merge(df_feats, on=['subject_id', 'lifelog_date'], how='left')

# 기존 데이터 추출 함수들 (생략 없이 사용)
def extract_activity(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        acts = grp['m_activity'].values
        h = grp['timestamp'].dt.hour.values
        for a in [0, 3, 4, 7, 8]: row[f'act_{a}_ratio'] = (acts == a).mean()
        row['act_active_ratio'] = ((acts == 7) | (acts == 8) | (acts == 3)).mean()
        row['act_still_ratio'] = (acts == 0).mean()
        row['act_n_records'] = len(acts)
        for seg, mask in [('morn', (h >= 6) & (h < 12)), ('aftn', (h >= 12) & (h < 18)),
                          ('eve', (h >= 18) & (h < 22)), ('night', (h >= 22) | (h < 6))]:
            s_acts = acts[mask]
            row[f'act_{seg}_active'] = ((s_acts == 7) | (s_acts == 8)).mean() if len(s_acts) > 0 else np.nan
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
        row['pedo_total_calories'] = grp['burned_calories'].sum()
        row['pedo_max_speed'] = grp['speed'].max()
        row['pedo_running_steps'] = grp['running_step'].sum()
        row['pedo_run_ratio'] = grp['running_step'].sum() / (grp['step'].sum() + 1)
        row['pedo_evening_steps'] = grp[grp['timestamp'].dt.hour.between(18, 21)]['step'].sum()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_screen(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        sc = grp['m_screen_use'].values
        h = grp['timestamp'].dt.hour.values
        row['screen_on_total'] = (sc > 0).sum()
        row['screen_on_ratio'] = (sc > 0).mean()
        for seg, mask in [('night', (h >= 22) | (h < 2)), ('eve', (h >= 20) & (h <= 23))]:
            s_sc = sc[mask]
            row[f'screen_{seg}_ratio'] = (s_sc > 0).mean() if len(s_sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_light(df_raw, col, prefix, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        vals = grp[col].dropna().values
        for k, v in agg_stats(vals, f'{prefix}_all').items(): row[k] = v
        feats.append(row)
    return pd.DataFrame(feats)

def extract_usage(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        total_time, late_time = 0, 0
        for ts, v in zip(grp['timestamp'], grp['m_usage_stats']):
            if isinstance(v, list):
                for app in v:
                    if isinstance(app, dict):
                        t = app.get('total_time', 0) or 0
                        total_time += t
                        if ts.hour >= 22 or ts.hour < 2: late_time += t
        row['usage_total_time'] = total_time
        row['usage_late_ratio'] = late_time / (total_time + 1)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_hr(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    df_m = df_raw[df_raw['timestamp'].dt.hour < 9].copy()
    feats = []
    for (sid, d), grp in df_m.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        all_v = []
        for ts, v in zip(grp['timestamp'], grp['heart_rate']):
            try: arr = np.asarray(v, dtype=float).ravel(); arr = arr[arr > 0]
            except Exception: arr = np.array([])
            all_v.extend(arr.tolist())
        sleep_hrs = np.array(all_v)
        sleep_hrs = sleep_hrs[sleep_hrs > 0] if len(sleep_hrs) > 0 else sleep_hrs
        for k, v in agg_stats(sleep_hrs, 'slp_hr').items(): row[k] = v
        row['slp_hr_deep_ratio'] = (sleep_hrs < 55).mean() if len(sleep_hrs) > 0 else np.nan
        row['slp_hr_awake_ratio'] = (sleep_hrs > 75).mean() if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 1:
            diffs = np.diff(sleep_hrs)
            row['slp_hr_rmssd'] = float(np.sqrt(np.nanmean(diffs ** 2)))
        else: row['slp_hr_rmssd'] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)

# ---------------------------------------------------------------------
# 🔥 [5순위 반영] Target Encoding 스무딩 (Smoothing) 추가
# ---------------------------------------------------------------------
def _build_subject_history(history_df, target):
    h = history_df[['subject_id', 'lifelog_date', target]].copy()
    h = h.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    hist = {}
    for sid, grp in h.groupby('subject_id'):
        hist[sid] = {'dates': grp['lifelog_date'].to_numpy(), 'labels': grp[target].to_numpy()}
    return hist

def _encode_from_history(history_map, query_df, windows, prior_mean, smoothing_weight=10.0):
    rows = []
    for sid, d in query_df[['subject_id', 'lifelog_date']].itertuples(index=False):
        if sid not in history_map:
            row = {'te_lag1': np.nan}
            for w in windows: row[f'te_enc{w}'] = prior_mean
            rows.append(row)
            continue
        
        dates = history_map[sid]['dates']
        labels = history_map[sid]['labels']
        k = np.searchsorted(dates, d, side='left')
        past = labels[:k]
        
        row = {'te_lag1': past[-1] if len(past) > 0 else np.nan}
        for w in windows:
            if len(past) > 0:
                window_past = past[-w:]
                n = len(window_past)
                mean_val = np.nanmean(window_past)
                # 데이터가 적을수록 prior_mean(전체 평균)에 가깝게 당기는 스무딩 수식
                row[f'te_enc{w}'] = (n * mean_val + smoothing_weight * prior_mean) / (n + smoothing_weight)
            else:
                row[f'te_enc{w}'] = prior_mean
        rows.append(row)
    return pd.DataFrame(rows, index=query_df.index)

def build_fold_safe_target_encoding(train_hist_df, tr_query_df, val_query_df, test_query_df, target, windows, prior_mean):
    history_map = _build_subject_history(train_hist_df, target)
    tr_te = _encode_from_history(history_map, tr_query_df, windows, prior_mean)
    val_te = _encode_from_history(history_map, val_query_df, windows, prior_mean)
    test_te = _encode_from_history(history_map, test_query_df, windows, prior_mean)
    return tr_te, val_te, test_te


# ---------------------------------------------------------------------
# Feature Table 구성
# ---------------------------------------------------------------------
def build_feature_table(train_df, sub_df):
    all_keys = pd.concat([train_df[['subject_id', 'lifelog_date']], sub_df[['subject_id', 'lifelog_date']]]).drop_duplicates().reset_index(drop=True)
    sleep_keys = pd.concat([train_df[['subject_id', 'sleep_date']], sub_df[['subject_id', 'sleep_date']]]).drop_duplicates().reset_index(drop=True)

    print('Extracting daytime features...')
    feat_dfs = []
    for name, fn, col, prefix in [
        ('mActivity', extract_activity, None, None), ('wPedo', extract_pedo, None, None),
        ('wHr', extract_hr, None, None), ('mScreenStatus', extract_screen, None, None),
        ('mLight', extract_light, 'm_light', 'mlight'), ('mUsageStats', extract_usage, None, None),
    ]:
        print(f'  {name}...')
        df = load_parquet(name)
        feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))
        del df; gc.collect()

    print('Extracting sleep-date features...')
    sleep_feat_dfs = []
    for name, fn in [('wHr', extract_sleep_hr)]:
        df = load_parquet(name)
        sleep_feat_dfs.append(fn(df, sleep_keys))
        del df; gc.collect()

    sleep_feats = sleep_feat_dfs[0]
    feat_all = feat_dfs[0]
    for df in feat_dfs[1:]: feat_all = feat_all.merge(df, on=['subject_id', 'lifelog_date'], how='outer')

    feat_all['dow'] = feat_all['lifelog_date'].dt.dayofweek
    feat_all['month'] = feat_all['lifelog_date'].dt.month
    feat_all['dow_sin'] = np.sin(2 * np.pi * feat_all['dow'] / 7)
    feat_all['dow_cos'] = np.cos(2 * np.pi * feat_all['dow'] / 7)

    feat_all = feat_all.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)

    # EWMA & Diff 트렌드 추가
    roll_cols = ['pedo_total_steps', 'screen_on_ratio', 'act_active_ratio', 'hr_daily_mean', 'usage_total_time']
    for col in roll_cols:
        if col not in feat_all.columns: continue
        g = feat_all.groupby('subject_id')[col]
        feat_all[f'{col}_lag1'] = g.shift(1)
        feat_all[f'{col}_diff1'] = feat_all[col] - feat_all[f'{col}_lag1'] 
        feat_all[f'{col}_ewma3'] = g.transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()) 
        feat_all[f'{col}_roll3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
    train_full = train_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    test_full = sub_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(feat_all, on=['subject_id', 'lifelog_date'], how='left')
    test_full = test_full.merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')

    feature_cols = [c for c in train_full.columns if c not in ['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS]
    return train_full, test_full, feature_cols


# ---------------------------------------------------------------------
# 🔥 [4순위 반영] Feature Selection 기준을 'gain'으로 변경
# ---------------------------------------------------------------------
def perform_feature_selection(train_full, feature_cols, targets, drop_ratio=0.15):
    print("\n[Feature Selection] Evaluating Feature Importance (Gain-based) to cut noise...")
    
    lgb_params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting_type': 'gbdt', 'learning_rate': 0.05,
        'n_estimators': 150, 'verbose': -1, 'n_jobs': -1,
        'random_state': 42,
        'importance_type': 'gain' # Split이 아닌 Gain(정보 획득량) 기준으로 변경!
    }
    if HAS_CUDA: lgb_params.update({'device': 'gpu'})
        
    importance_df = pd.DataFrame({'feature': feature_cols})
    importance_df['importance'] = 0.0
    
    for t in targets:
        y = train_full[t].values
        model = lgb.LGBMClassifier(**lgb_params)
        try: model.fit(train_full[feature_cols], y)
        except Exception:
            cpu_params = dict(lgb_params); cpu_params['device'] = 'cpu'
            model = lgb.LGBMClassifier(**cpu_params)
            model.fit(train_full[feature_cols], y)
            
        importance_df['importance'] += model.feature_importances_ / len(targets)
        
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    n_drop = int(len(feature_cols) * drop_ratio)
    drop_features = importance_df.tail(n_drop)['feature'].tolist()
    zero_imp_features = importance_df[importance_df['importance'] == 0]['feature'].tolist()
    
    final_drop_list = list(set(drop_features + zero_imp_features))
    
    # 중요도 순으로 정렬된 상태를 유지하며 반환 (Level 1 스태킹에 활용하기 위함)
    selected_features = importance_df[~importance_df['feature'].isin(final_drop_list)]['feature'].tolist()
    
    print(f"  - Original features: {len(feature_cols)}개")
    print(f"  - Dropped noise features: {len(final_drop_list)}개")
    print(f"  - Selected elite features: {len(selected_features)}개")
    
    return selected_features


# ---------------------------------------------------------------------
# 🔥 [2순위 & 3순위 반영] 누수 차단 Optuna + LGBM Meta Stacking
# ---------------------------------------------------------------------
def train_and_predict(train_full, test_full, feature_cols):
    X_train_base = train_full[feature_cols].copy()
    X_test_base = test_full[feature_cols].copy()

    seeds = [42, 1234, 9999, 7, 314, 2025, 777, 555]
    n_folds = 5
    n_optuna_trials = 50 # 시행 횟수 50회로 상향!
    te_windows = [3, 7, 14, 21]

    # Level 1 메타 모델 (LightGBM으로 교체)
    meta_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4,
        'n_estimators': 300, 'verbose': -1, 'random_state': 42
    }
    if HAS_CUDA: meta_params['device'] = 'gpu'

    oof_preds = np.zeros((len(X_train_base), len(TARGETS)))
    test_preds = np.zeros((len(X_test_base), len(TARGETS)))
    
    # Meta Input을 위해 상위 10개의 오리지널 피처 추출
    top_k_features = feature_cols[:10] 

    for ti, target in enumerate(TARGETS):
        y = train_full[target].values
        print(f'\n{"="*50}\n=== Target: {target} | pos_rate: {y.mean():.3f} ===\n{"="*50}')

        # -----------------------------------------------------
        # 🎯 [2순위 적용] 데이터 누수 없는 완벽한 Optuna 튜닝
        # -----------------------------------------------------
        print(f"  [Optuna] Tuning on strict Train-Split to prevent Leakage (Trials: {n_optuna_trials})...")
        
        # 전체 데이터 대상이 아닌, 5-Fold 중 Fold 0의 훈련용(tr_idx) 데이터만 추출하여 튜닝
        outer_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        tune_tr_idx, _ = next(outer_skf.split(X_train_base, y))
        X_tune = X_train_base.iloc[tune_tr_idx].copy()
        y_tune = y[tune_tr_idx]

        tune_inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def objective_lgb(trial):
            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
                'n_estimators': 300, 'verbose': -1, 'n_jobs': -1, 'random_state': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9)
            }
            if HAS_CUDA: params.update({'device': 'gpu'})
            
            cv_loss = []
            for t_idx, v_idx in tune_inner_skf.split(X_tune, y_tune):
                model = lgb.LGBMClassifier(**params)
                try: model.fit(X_tune.iloc[t_idx], y_tune[t_idx])
                except: 
                    params['device'] = 'cpu'
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_tune.iloc[t_idx], y_tune[t_idx])
                preds = model.predict_proba(X_tune.iloc[v_idx])[:, 1]
                cv_loss.append(log_loss(y_tune[v_idx], preds))
            return np.mean(cv_loss)

        def objective_xgb(trial):
            params = {
                'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 300, 'n_jobs': -1, 'random_state': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9)
            }
            if HAS_CUDA: params.update({'tree_method': 'hist', 'device': 'cuda'})
            cv_loss = []
            for t_idx, v_idx in tune_inner_skf.split(X_tune, y_tune):
                model = xgb.XGBClassifier(**params)
                model.fit(X_tune.iloc[t_idx], y_tune[t_idx], verbose=False)
                preds = model.predict_proba(X_tune.iloc[v_idx])[:, 1]
                cv_loss.append(log_loss(y_tune[v_idx], preds))
            return np.mean(cv_loss)

        def objective_cat(trial):
            params = {
                'loss_function': 'Logloss', 'iterations': 300, 'verbose': False, 'thread_count': -1, 'random_seed': 42,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 8)
            }
            if HAS_CUDA: params.update({'task_type': 'GPU'})
            cv_loss = []
            for t_idx, v_idx in tune_inner_skf.split(X_tune, y_tune):
                model = CatBoostClassifier(**params)
                model.fit(X_tune.iloc[t_idx], y_tune[t_idx])
                preds = model.predict_proba(X_tune.iloc[v_idx])[:, 1]
                cv_loss.append(log_loss(y_tune[v_idx], preds))
            return np.mean(cv_loss)

        # 튜닝 실행
        study_lgb = optuna.create_study(direction='minimize'); study_lgb.optimize(objective_lgb, n_trials=n_optuna_trials)
        study_xgb = optuna.create_study(direction='minimize'); study_xgb.optimize(objective_xgb, n_trials=n_optuna_trials)
        study_cat = optuna.create_study(direction='minimize'); study_cat.optimize(objective_cat, n_trials=n_optuna_trials)

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
        # ⚔️ [LEVEL 0] 최적 파라미터로 본학습
        # -----------------------------------------------------
        print("  [Level 0] Training Models...")
        target_oof_lgb, target_oof_xgb, target_oof_cat = np.zeros(len(X_train_base)), np.zeros(len(X_train_base)), np.zeros(len(X_train_base))
        target_test_lgb, target_test_xgb, target_test_cat = np.zeros(len(X_test_base)), np.zeros(len(X_test_base)), np.zeros(len(X_test_base))

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
                    
                    prior_mean = y_tr.mean() # 5순위: 스무딩에 쓰일 전체 평균 전달
                    tr_te, val_te, test_te = build_fold_safe_target_encoding(hist_df, tr_query, val_query, test_query, target, te_windows, prior_mean)
                    X_tr = pd.concat([X_tr.reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
                    X_val = pd.concat([X_val.reset_index(drop=True), val_te.reset_index(drop=True)], axis=1)
                    X_te = pd.concat([X_te.reset_index(drop=True), test_te.reset_index(drop=True)], axis=1)

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

        print(f"  [Level 0] LogLoss - LGBM: {log_loss(y, target_oof_lgb):.4f}, XGB: {log_loss(y, target_oof_xgb):.4f}, CAT: {log_loss(y, target_oof_cat):.4f}")

        # -----------------------------------------------------
        # 🔗 [3순위 적용] LightGBM 기반 강력한 메타 모델 스태킹
        # -----------------------------------------------------
        print("  [Level 1] Training LightGBM Meta-Model with Probabilities + Top Raw Features...")
        
        # 3개 확률 변수에 더해 상위 10개의 핵심 원본 피처를 합쳐 비선형 시너지 극대화
        X_meta_train = np.column_stack([target_oof_lgb, target_oof_xgb, target_oof_cat, train_full[top_k_features].values])
        X_meta_test = np.column_stack([target_test_lgb, target_test_xgb, target_test_cat, test_full[top_k_features].values])

        meta_oof, meta_test = np.zeros(len(X_train_base)), np.zeros(len(X_test_base))
        meta_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for meta_tr_idx, meta_val_idx in meta_skf.split(X_meta_train, y):
            meta_model = lgb.LGBMClassifier(**meta_params)
            
            try: meta_model.fit(X_meta_train[meta_tr_idx], y[meta_tr_idx])
            except: 
                cpu_meta = dict(meta_params); cpu_meta['device'] = 'cpu'
                meta_model = lgb.LGBMClassifier(**cpu_meta)
                meta_model.fit(X_meta_train[meta_tr_idx], y[meta_tr_idx])
                
            meta_oof[meta_val_idx] = meta_model.predict_proba(X_meta_train[meta_val_idx])[:, 1]
            meta_test += meta_model.predict_proba(X_meta_test)[:, 1] / n_folds

        oof_preds[:, ti], test_preds[:, ti] = meta_oof, meta_test
        print(f'  🎯 [Level 1] Final Stacked OOF [{target}]: {log_loss(y, oof_preds[:, ti]):.4f}')

    return oof_preds, test_preds

def write_report(report_data):
    lines = ['=' * 80, 'Baseline Master v13 run report', f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    lines.extend([f"  Total OOF: {report_data['avg_oof']:.4f}", f"  Feature count: {report_data['n_features']}"])
    with open(REPORT_PATH, 'w', encoding='utf-8') as f: f.write('\n'.join(lines))
    print('\n' + '\n'.join(lines))

def main():
    ensure_dirs()
    print('Starting Master Training Pipeline v13...')
    
    train_df = pd.read_csv(TRAIN_PATH)
    sub_df = pd.read_csv(SUB_PATH)
    for df in [train_df, sub_df]:
        df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
        df['sleep_date'] = pd.to_datetime(df['sleep_date'])

    train_full, test_full, feature_cols = build_feature_table(train_df, sub_df)
    
    # 2. Gain 기반 피처 컷다운 및 중요도 정렬
    elite_feature_cols = perform_feature_selection(train_full, feature_cols, TARGETS, drop_ratio=0.15)
    
    # 3. 누수 없는 Optuna + LGBM 메타 스태킹
    oof_preds, test_preds = train_and_predict(train_full, test_full, elite_feature_cols)

    per_target = {t: log_loss(train_full[t].values, oof_preds[:, i]) for i, t in enumerate(TARGETS)}
    oof_total = float(np.mean(list(per_target.values())))

    print(f'\n{"=" * 55}\nMaster v13 Total OOF: {oof_total:.4f}\n{"=" * 55}')

    submission = sub_df[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    for i, t in enumerate(TARGETS): submission[t] = test_preds[:, i].clip(0.02, 0.98)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f'submission saved: {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
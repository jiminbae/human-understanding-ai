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
# v8 Master: Rank Average Ensemble + Deep Optuna + Safe Stacking
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

EXP_TAG = '_master_v8_hybrid'
OUTPUT_PATH = OUTPUT_DIR / f'submission{EXP_TAG}.csv'
REPORT_PATH = REPORT_DIR / f'report{EXP_TAG}.txt'

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

def safe_mean(vals): return np.nanmean(np.array(vals)) if len(vals) > 0 else np.nan

def load_parquet(name):
    df = pd.read_parquet(DATA_DIR / f'ch2025_{name}.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- Feature Extraction Functions (기존 유지) ---
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
        feats.append(row)
    return pd.DataFrame(feats)

def extract_pedo(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row['pedo_total_steps'] = grp['step'].sum()
        row['pedo_total_distance'] = grp['distance'].sum()
        row['pedo_mean_speed'] = grp['speed'].mean()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_hr(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        vals = []
        for v in grp['heart_rate']:
            try: vals.extend(np.asarray(v, dtype=float).ravel().tolist())
            except: pass
        arr = np.array(vals)
        arr = arr[arr > 0]
        row['hr_daily_mean'] = np.nanmean(arr) if len(arr) > 0 else np.nan
        feats.append(row)
    return keys.merge(pd.DataFrame(feats), on=['subject_id', 'lifelog_date'], how='left')

def extract_screen(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row['screen_on_ratio'] = (grp['m_screen_use'].values > 0).mean()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_light(df_raw, col, prefix, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row[f'{prefix}_all_mean'] = safe_mean(grp[col].dropna().values)
        feats.append(row)
    return pd.DataFrame(feats)

def extract_ac(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        row['ac_charging_ratio'] = grp['m_charging'].mean()
        feats.append(row)
    return pd.DataFrame(feats)

def extract_usage(df_raw, keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'lifelog_date': d}
        total_time = 0
        for val in grp['m_usage_stats']:
            if isinstance(val, list):
                for app in val:
                    if isinstance(app, dict): total_time += app.get('total_time', 0)
        row['usage_total_time'] = total_time
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_hr(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    df_m = df_raw[df_raw['timestamp'].dt.hour < 9].copy()
    feats = []
    for (sid, d), grp in df_m.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        vals = []
        for v in grp['heart_rate']:
            try: vals.extend(np.asarray(v, dtype=float).ravel().tolist())
            except: pass
        arr = np.array(vals)
        arr = arr[arr > 0]
        row['slp_hr_mean'] = np.nanmean(arr) if len(arr) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)

def extract_sleep_pedo(df_raw, sleep_keys):
    df_raw['date'] = df_raw['timestamp'].dt.normalize()
    feats = []
    for (sid, d), grp in df_raw.groupby(['subject_id', 'date']):
        row = {'subject_id': sid, 'sleep_date': d}
        row['slp_pedo_steps'] = grp[grp['timestamp'].dt.hour < 9]['step'].sum()
        feats.append(row)
    return pd.DataFrame(feats)

# --- 신규 편입: Rank Average 함수 ---
def rank_average(arrays):
    """예측 확률을 순위(Rank)로 변환 후 평균을 내어 스케일 왜곡을 방지"""
    n = len(arrays[0])
    ranks = []
    for arr in arrays:
        r = np.argsort(np.argsort(arr)).astype(float) / (n - 1)
        ranks.append(r)
    return np.mean(ranks, axis=0)

def _build_subject_history(history_df, target):
    h = history_df[['subject_id', 'lifelog_date', target]].copy()
    h = h.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)
    hist = {}
    for sid, grp in h.groupby('subject_id'): hist[sid] = {'dates': grp['lifelog_date'].to_numpy(), 'labels': grp[target].to_numpy()}
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
        for w in windows: row[f'te_enc{w}'] = np.nanmean(past[-w:]) if len(past) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=query_df.index)

def build_fold_safe_target_encoding(train_hist_df, tr_query_df, val_query_df, test_query_df, target, windows):
    history_map = _build_subject_history(train_hist_df, target)
    return _encode_from_history(history_map, tr_query_df, windows), _encode_from_history(history_map, val_query_df, windows), _encode_from_history(history_map, test_query_df, windows)

def build_feature_table(train_df, sub_df):
    all_keys = pd.concat([train_df[['subject_id', 'lifelog_date']], sub_df[['subject_id', 'lifelog_date']]]).drop_duplicates().reset_index(drop=True)
    sleep_keys = pd.concat([train_df[['subject_id', 'sleep_date']], sub_df[['subject_id', 'sleep_date']]]).drop_duplicates().reset_index(drop=True)

    feat_dfs = []
    for name, fn, col, prefix in [
        ('mActivity', extract_activity, None, None), ('wPedo', extract_pedo, None, None),
        ('wHr', extract_hr, None, None), ('mScreenStatus', extract_screen, None, None),
        ('mLight', extract_light, 'm_light', 'mlight'), ('wLight', extract_light, 'w_light', 'wlight'),
        ('mACStatus', extract_ac, None, None), ('mUsageStats', extract_usage, None, None)
    ]:
        df = load_parquet(name)
        feat_dfs.append(fn(df, col, prefix, all_keys) if col else fn(df, all_keys))
        del df; gc.collect()

    sleep_feat_dfs = []
    for name, fn in [('wHr', extract_sleep_hr), ('wPedo', extract_sleep_pedo)]:
        df = load_parquet(name)
        sleep_feat_dfs.append(fn(df, sleep_keys))
        del df; gc.collect()

    sleep_feats = sleep_feat_dfs[0]
    for df in sleep_feat_dfs[1:]: sleep_feats = sleep_feats.merge(df, on=['subject_id', 'sleep_date'], how='outer')

    feat_all = feat_dfs[0]
    for df in feat_dfs[1:]: feat_all = feat_all.merge(df, on=['subject_id', 'lifelog_date'], how='outer')

    feat_all['dow'] = feat_all['lifelog_date'].dt.dayofweek
    feat_all['month'] = feat_all['lifelog_date'].dt.month
    feat_all['is_weekend'] = (feat_all['dow'] >= 5).astype(int)

    feat_all = feat_all.sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)

    train_full = train_df.merge(feat_all, on=['subject_id', 'lifelog_date'], how='left').merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')
    test_full = sub_df[['subject_id', 'lifelog_date', 'sleep_date']].merge(feat_all, on=['subject_id', 'lifelog_date'], how='left').merge(sleep_feats, on=['subject_id', 'sleep_date'], how='left')

    feature_cols = [c for c in train_full.columns if c not in ['subject_id', 'lifelog_date', 'sleep_date'] + TARGETS]
    return train_full, test_full, feature_cols

def perform_feature_selection(train_full, feature_cols, targets, drop_ratio=0.15):
    print("\n[Feature Selection] Evaluating Feature Importance (Gain-based)...")
    lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'learning_rate': 0.05, 'n_estimators': 150, 'verbose': -1, 'n_jobs': -1, 'random_state': 42, 'importance_type': 'gain'}
    if HAS_CUDA: lgb_params.update({'device': 'gpu'})
        
    importance_df = pd.DataFrame({'feature': feature_cols})
    importance_df['importance'] = 0.0
    
    for t in targets:
        model = lgb.LGBMClassifier(**lgb_params)
        try: model.fit(train_full[feature_cols], train_full[t].values)
        except: 
            cpu_params = dict(lgb_params); cpu_params['device'] = 'cpu'
            model = lgb.LGBMClassifier(**cpu_params)
            model.fit(train_full[feature_cols], train_full[t].values)
        importance_df['importance'] += model.feature_importances_ / len(targets)
        
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    n_drop = int(len(feature_cols) * drop_ratio)
    drop_features = importance_df.tail(n_drop)['feature'].tolist()
    zero_imp_features = importance_df[importance_df['importance'] == 0]['feature'].tolist()
    
    final_drop_list = list(set(drop_features + zero_imp_features))
    selected_features = [c for c in feature_cols if c not in final_drop_list]
    print(f"  - Selected elite features: {len(selected_features)} / {len(feature_cols)}")
    return selected_features

def train_and_predict(train_full, test_full, feature_cols):
    X_train_base = train_full[feature_cols].copy()
    X_test_base = test_full[feature_cols].copy()

    # --- 전략 3: 연산 자원 극대화 (Optuna 100회, 시드 12개) ---
    n_optuna_trials = 100 
    seeds = [42, 1234, 9999, 7, 314, 2025, 777, 555, 1111, 2222, 3333, 4444] 
    n_folds = 5
    te_windows = [3, 7, 14, 21]

    # --- 전략 4: 안전한 메타 스태킹 ---
    meta_params = {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000}

    oof_preds = np.zeros((len(X_train_base), len(TARGETS)))
    test_preds = np.zeros((len(X_test_base), len(TARGETS)))

    for ti, target in enumerate(TARGETS):
        y = train_full[target].values
        print(f'\n{"="*40}\n=== Target: {target} | pos_rate: {y.mean():.3f} ===\n{"="*40}')

        print(f"  [Optuna] Deep Searching for Golden Parameters (Trials: {n_optuna_trials})...")
        _tune_outer_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        _tune_tr_idx, _ = next(iter(_tune_outer_skf.split(X_train_base, y)))
        X_tune = X_train_base.iloc[_tune_tr_idx].reset_index(drop=True)
        y_tune = y[_tune_tr_idx]

        tune_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 

        def objective_lgb(trial):
            params = {'objective': 'binary', 'n_estimators': 300, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 'num_leaves': trial.suggest_int('num_leaves', 15, 63), 'verbose': -1, 'n_jobs': -1}
            if HAS_CUDA: params['device'] = 'gpu'
            cv_loss = []
            for tr_idx, val_idx in tune_skf.split(X_tune, y_tune):
                model = lgb.LGBMClassifier(**params)
                try: model.fit(X_tune.iloc[tr_idx], y_tune[tr_idx])
                except: params['device']='cpu'; model = lgb.LGBMClassifier(**params); model.fit(X_tune.iloc[tr_idx], y_tune[tr_idx])
                cv_loss.append(log_loss(y_tune[val_idx], model.predict_proba(X_tune.iloc[val_idx])[:, 1]))
            return np.mean(cv_loss)

        def objective_xgb(trial):
            params = {'objective': 'binary:logistic', 'n_estimators': 300, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 'max_depth': trial.suggest_int('max_depth', 3, 8), 'n_jobs': -1}
            if HAS_CUDA: params.update({'tree_method': 'hist', 'device': 'cuda'})
            cv_loss = []
            for tr_idx, val_idx in tune_skf.split(X_tune, y_tune):
                model = xgb.XGBClassifier(**params)
                model.fit(X_tune.iloc[tr_idx], y_tune[tr_idx], verbose=False)
                cv_loss.append(log_loss(y_tune[val_idx], model.predict_proba(X_tune.iloc[val_idx])[:, 1]))
            return np.mean(cv_loss)

        def objective_cat(trial):
            params = {'loss_function': 'Logloss', 'iterations': 300, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 'depth': trial.suggest_int('depth', 4, 8), 'verbose': False, 'thread_count': -1}
            if HAS_CUDA: params['task_type'] = 'GPU'
            cv_loss = []
            for tr_idx, val_idx in tune_skf.split(X_tune, y_tune):
                model = CatBoostClassifier(**params)
                model.fit(X_tune.iloc[tr_idx], y_tune[tr_idx])
                cv_loss.append(log_loss(y_tune[val_idx], model.predict_proba(X_tune.iloc[val_idx])[:, 1]))
            return np.mean(cv_loss)

        study_lgb = optuna.create_study(direction='minimize'); study_lgb.optimize(objective_lgb, n_trials=n_optuna_trials)
        study_xgb = optuna.create_study(direction='minimize'); study_xgb.optimize(objective_xgb, n_trials=n_optuna_trials)
        study_cat = optuna.create_study(direction='minimize'); study_cat.optimize(objective_cat, n_trials=n_optuna_trials)

        best_lgb_params = {**study_lgb.best_params, 'objective': 'binary', 'n_estimators': 2000, 'verbose': -1, 'n_jobs': -1}
        best_xgb_params = {**study_xgb.best_params, 'objective': 'binary:logistic', 'n_estimators': 2000, 'n_jobs': -1}
        best_cat_params = {**study_cat.best_params, 'loss_function': 'Logloss', 'iterations': 2000, 'verbose': False, 'thread_count': -1}
        if HAS_CUDA:
            best_lgb_params['device'] = 'gpu'; best_xgb_params.update({'tree_method': 'hist', 'device': 'cuda'}); best_cat_params['task_type'] = 'GPU'
        else: best_lgb_params['device'] = 'cpu'

        print("  [Level 0] Training Main Ensemble with Expanded Seeds...")
        target_oof_lgb, target_oof_xgb, target_oof_cat = np.zeros(len(X_train_base)), np.zeros(len(X_train_base)), np.zeros(len(X_train_base))
        
        # --- 시드별 예측값을 따로 담아둘 리스트 생성 ---
        seed_test_lgb_list, seed_test_xgb_list, seed_test_cat_list = [], [], []

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
                except: cpu_params = dict(best_lgb_params); cpu_params['device']='cpu'; model_lgb = lgb.LGBMClassifier(**{**cpu_params, 'random_state': seed}); model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
                model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)

                seed_oof_lgb[val_idx], seed_oof_xgb[val_idx], seed_oof_cat[val_idx] = model_lgb.predict_proba(X_val)[:, 1], model_xgb.predict_proba(X_val)[:, 1], model_cat.predict_proba(X_val)[:, 1]
                seed_test_lgb += model_lgb.predict_proba(X_te)[:, 1] / n_folds
                seed_test_xgb += model_xgb.predict_proba(X_te)[:, 1] / n_folds
                seed_test_cat += model_cat.predict_proba(X_te)[:, 1] / n_folds

            target_oof_lgb += seed_oof_lgb / len(seeds); target_oof_xgb += seed_oof_xgb / len(seeds); target_oof_cat += seed_oof_cat / len(seeds)
            seed_test_lgb_list.append(seed_test_lgb); seed_test_xgb_list.append(seed_test_xgb); seed_test_cat_list.append(seed_test_cat)

        # --- 전략 4 적용: 시드별 예측값을 Rank Average로 병합하여 안정성 극대화 ---
        target_test_lgb = rank_average(seed_test_lgb_list)
        target_test_xgb = rank_average(seed_test_xgb_list)
        target_test_cat = rank_average(seed_test_cat_list)

        print("  [Level 1] Training Meta-Model (Safe Stacking)...")
        X_meta_train = np.column_stack([target_oof_lgb, target_oof_xgb, target_oof_cat])
        X_meta_test = np.column_stack([target_test_lgb, target_test_xgb, target_test_cat])

        meta_oof, meta_test = np.zeros(len(X_train_base)), np.zeros(len(X_test_base))
        meta_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for meta_tr_idx, meta_val_idx in meta_skf.split(X_meta_train, y):
            meta_model = LogisticRegression(**meta_params)
            meta_model.fit(X_meta_train[meta_tr_idx], y[meta_tr_idx])
            meta_oof[meta_val_idx] = meta_model.predict_proba(X_meta_train[meta_val_idx])[:, 1]
            meta_test += meta_model.predict_proba(X_meta_test)[:, 1] / n_folds

        oof_preds[:, ti], test_preds[:, ti] = meta_oof, meta_test
        print(f'  🎯 [Level 1] Final Stacked OOF [{target}]: {log_loss(y, oof_preds[:, ti]):.4f}')

    return oof_preds, test_preds

def main():
    ensure_dirs()
    print('Starting Hybrid Pipeline (v8 Master)...')
    
    train_df = pd.read_csv(TRAIN_PATH)
    sub_df = pd.read_csv(SUB_PATH)
    for df in [train_df, sub_df]: df['lifelog_date'] = pd.to_datetime(df['lifelog_date']); df['sleep_date'] = pd.to_datetime(df['sleep_date'])

    train_full, test_full, feature_cols = build_feature_table(train_df, sub_df)
    elite_feature_cols = perform_feature_selection(train_full, feature_cols, TARGETS, drop_ratio=0.15)
    oof_preds, test_preds = train_and_predict(train_full, test_full, elite_feature_cols)

    oof_total = float(np.mean([log_loss(train_full[t].values, oof_preds[:, i]) for i, t in enumerate(TARGETS)]))
    print(f'\n{"=" * 55}\nMaster v8 Hybrid Total OOF: {oof_total:.4f}\n{"=" * 55}')

    submission = sub_df[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    for i, t in enumerate(TARGETS): submission[t] = test_preds[:, i].clip(0.02, 0.98)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f'submission saved: {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
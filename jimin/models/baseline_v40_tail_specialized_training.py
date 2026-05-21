# v40: tail-specialized training on top of the public-validated v34 anchor.
#   - v38 proved that final tail blocks need a different policy than interior rows.
#   - v40 stops borrowing old tail predictions and trains fresh tail-only models.
#   - Validation mirrors the real tail structure: the last hidden block per subject
#     is held out, and target encodings use only visible past labels.
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
import optuna

try:
    import baseline_v33_long_history_cross_target as v33
except ModuleNotFoundError:
    from jimin.models import baseline_v33_long_history_cross_target as v33


optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TAIL_TARGETS = [
    t.strip()
    for t in os.environ.get('V40_TAIL_TARGETS', 'Q2,S1,S2,S3,S4').split(',')
    if t.strip()
]
TE_WINDOWS = [
    int(x.strip())
    for x in os.environ.get('V40_TE_WINDOWS', '3,7,14,21,30,45,60').split(',')
    if x.strip()
]
SEEDS = [
    int(x.strip())
    for x in os.environ.get('V40_SEEDS', '42,1234,9999,7,314,2025').split(',')
    if x.strip()
]
N_OPTUNA_TRIALS = int(os.environ.get('V40_OPTUNA_TRIALS', '30'))
MAX_STABLE_FEATURES = int(os.environ.get('V40_MAX_FEATURES', str(v33.MAX_STABLE_FEATURES)))
STABILITY_THRESHOLD = float(os.environ.get('V40_STAB_THRESHOLD', str(v33.STABILITY_THRESHOLD)))
RECENCY_ALPHA = float(os.environ.get('V40_RECENCY_ALPHA', '0.35'))

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

EXP_TAG = os.environ.get('V40_EXP_TAG', 'v40_tail_specialized_training')
ANCHOR_OOF_PATH = OOF_DIR / 'oof_v35_winning_policy_ablation_q1p1_q3s4p2.csv'
ANCHOR_SUB_PATH = SUB_DIR / 'submission_v35_winning_policy_ablation_q1p1_q3s4p2.csv'


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
    for path in [SUB_DIR, OOF_DIR, LOG_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['lifelog_date', 'sleep_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def build_actual_tail_mask(train, sub):
    tail_mask = np.zeros(len(sub), dtype=bool)
    for sid, grp in sub.groupby('subject_id', sort=True):
        train_dates = train.loc[train['subject_id'] == sid, 'sleep_date']
        for idx, sleep_date in grp['sleep_date'].items():
            tail_mask[idx] = not bool((train_dates > sleep_date).any())
    return pd.Series(tail_mask, index=sub.index)


def build_proxy_tail_mask(train, test, min_visible_tail=8):
    tail = pd.Series(False, index=train.index)
    tail_lengths = {}
    for sid, grp in train.groupby('subject_id', sort=True):
        combined = pd.concat([
            train.loc[train['subject_id'] == sid, ['sleep_date']].assign(kind='T'),
            test.loc[test['subject_id'] == sid, ['sleep_date']].assign(kind='X'),
        ]).sort_values('sleep_date')
        runs = []
        for row in combined.itertuples(index=False):
            if not runs or runs[-1][0] != row.kind:
                runs.append([row.kind, 1])
            else:
                runs[-1][1] += 1
        x_runs = [n for kind, n in runs if kind == 'X']

        idx = grp.sort_values('sleep_date').index.to_numpy()
        tail_len = min(x_runs[-1], max(0, len(idx) - min_visible_tail))
        tail_lengths[sid] = int(tail_len)
        if tail_len > 0:
            tail.loc[idx[-tail_len:]] = True
    return tail, tail_lengths


def add_visible_history_shape_features(train_full, test_full, proxy_tail_mask):
    train_out = train_full.copy()
    test_out = test_full.copy()

    train_out['v40_hist_rank'] = 0.0
    train_out['v40_hist_frac'] = 0.0
    train_out['v40_days_from_first'] = 0.0
    train_out['v40_days_since_visible_last'] = np.nan

    test_out['v40_hist_rank'] = 0.0
    test_out['v40_hist_frac'] = 0.0
    test_out['v40_days_from_first'] = 0.0
    test_out['v40_days_since_visible_last'] = np.nan

    visible_train = train_full.loc[~proxy_tail_mask].copy()
    for sid, grp in train_full.groupby('subject_id', sort=True):
        ordered = grp.sort_values('sleep_date')
        visible = visible_train.loc[visible_train['subject_id'] == sid].sort_values('sleep_date')
        visible_dates = visible['sleep_date'].tolist()
        first_date = ordered['sleep_date'].min()
        denom = max(1, len(ordered) - 1)

        for rank, (idx, row) in enumerate(ordered.iterrows()):
            train_out.loc[idx, 'v40_hist_rank'] = float(rank)
            train_out.loc[idx, 'v40_hist_frac'] = float(rank / denom)
            train_out.loc[idx, 'v40_days_from_first'] = float((row['sleep_date'] - first_date).days)
            past_visible = [d for d in visible_dates if d < row['sleep_date']]
            if past_visible:
                train_out.loc[idx, 'v40_days_since_visible_last'] = float(
                    (row['sleep_date'] - past_visible[-1]).days
                )

        test_grp = test_full.loc[test_full['subject_id'] == sid].sort_values('sleep_date')
        actual_visible_dates = train_full.loc[
            train_full['subject_id'] == sid, 'sleep_date'
        ].sort_values().tolist()
        all_dates = pd.concat([
            ordered[['sleep_date']],
            test_grp[['sleep_date']],
        ]).sort_values('sleep_date')['sleep_date'].tolist()
        first_all_date = min(all_dates)
        denom_all = max(1, len(all_dates) - 1)
        for rank, (idx, row) in enumerate(
            pd.concat([
                ordered[['sleep_date']].assign(kind='train', idx=ordered.index),
                test_grp[['sleep_date']].assign(kind='test', idx=test_grp.index),
            ]).sort_values('sleep_date').iterrows()
        ):
            if row['kind'] != 'test':
                continue
            test_idx = int(row['idx'])
            test_out.loc[test_idx, 'v40_hist_rank'] = float(rank)
            test_out.loc[test_idx, 'v40_hist_frac'] = float(rank / denom_all)
            test_out.loc[test_idx, 'v40_days_from_first'] = float(
                (row['sleep_date'] - first_all_date).days
            )
            past_visible = [d for d in actual_visible_dates if d < row['sleep_date']]
            if past_visible:
                test_out.loc[test_idx, 'v40_days_since_visible_last'] = float(
                    (row['sleep_date'] - past_visible[-1]).days
                )
    new_cols = [
        'v40_hist_rank',
        'v40_hist_frac',
        'v40_days_from_first',
        'v40_days_since_visible_last',
    ]
    return train_out, test_out, new_cols


def add_anchor_cross_features(train_x, test_x, anchor_oof, anchor_sub, target):
    train_out = train_x.copy()
    test_out = test_x.copy()
    cols = []
    for other in TARGETS:
        if other == target:
            continue
        col = f'v40_anchor_{other}'
        train_out[col] = anchor_oof[other].values
        test_out[col] = anchor_sub[other].values
        cols.append(col)
    return train_out, test_out, cols


def add_forward_only_te(train_x, test_x, history_df, target):
    history_map = v33._build_subject_history(history_df, target)
    train_te = v33._encode_from_history(
        history_map,
        train_x[['subject_id', 'lifelog_date']],
        TE_WINDOWS,
    ).reset_index(drop=True)
    test_te = v33._encode_from_history(
        history_map,
        test_x[['subject_id', 'lifelog_date']],
        TE_WINDOWS,
    ).reset_index(drop=True)
    train_out = pd.concat([train_x.reset_index(drop=True), train_te], axis=1)
    test_out = pd.concat([test_x.reset_index(drop=True), test_te], axis=1)
    return train_out, test_out, train_te.columns.tolist()


def build_recency_weights(frame):
    weights = np.ones(len(frame), dtype=float)
    if RECENCY_ALPHA <= 0:
        return weights
    rank = (
        frame.sort_values(['subject_id', 'sleep_date'])
        .groupby('subject_id')
        .cumcount()
        .reindex(frame.sort_values(['subject_id', 'sleep_date']).index)
    )
    max_rank = rank.groupby(frame.loc[rank.index, 'subject_id']).transform('max').replace(0, 1)
    normalized = (rank / max_rank).reindex(frame.index).fillna(0.0).to_numpy(dtype=float)
    return 1.0 + RECENCY_ALPHA * normalized


def select_stable_features(x_train, y_train):
    nonzero_rate, mean_imp = v33.compute_stability_scores(x_train, y_train)
    stable_idx = np.where(nonzero_rate >= STABILITY_THRESHOLD)[0]
    stable_idx = stable_idx[np.argsort(mean_imp[stable_idx])[::-1]][:MAX_STABLE_FEATURES]
    if len(stable_idx) == 0:
        return x_train.columns.tolist()
    return x_train.columns[stable_idx].tolist()


def build_model_params():
    return {
        'lgb': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 300,
            'verbose': -1,
            'n_jobs': -1,
        },
        'xgb': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': 300,
            'n_jobs': -1,
        },
        'cat': {
            'loss_function': 'Logloss',
            'iterations': 300,
            'verbose': False,
            'thread_count': -1,
        },
    }


def tune_params(x_train, y_train, x_val, y_val, sample_weight):
    base = build_model_params()

    def objective_lgb(trial):
        params = {
            **base['lgb'],
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        }
        if v33.HAS_CUDA:
            params.update({'device': 'gpu'})
        model = lgb.LGBMClassifier(**params)
        try:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        except Exception:
            params['device'] = 'cpu'
            model = lgb.LGBMClassifier(**params)
            model.fit(x_train, y_train, sample_weight=sample_weight)
        return log_loss(y_val, model.predict_proba(x_val)[:, 1])

    def objective_xgb(trial):
        params = {
            **base['xgb'],
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        }
        if v33.HAS_CUDA:
            params.update({'tree_method': 'hist', 'device': 'cuda'})
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
        return log_loss(y_val, model.predict_proba(x_val)[:, 1])

    def objective_cat(trial):
        params = {
            **base['cat'],
            'random_seed': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        }
        if v33.HAS_CUDA:
            params.update({'task_type': 'GPU'})
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return log_loss(y_val, model.predict_proba(x_val)[:, 1])

    studies = {}
    for name, objective in [('lgb', objective_lgb), ('xgb', objective_xgb), ('cat', objective_cat)]:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS)
        studies[name] = study
    return studies


def fit_validation_ensemble(
    x_train,
    y_train,
    x_val,
    y_val,
    sample_weight,
    studies,
):
    val_preds = {name: np.zeros(len(x_val)) for name in ['lgb', 'xgb', 'cat']}
    best_rounds = {name: [] for name in ['lgb', 'xgb', 'cat']}

    for seed in SEEDS:
        lgb_params = {
            **studies['lgb'].best_params,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 2000,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': seed,
        }
        xgb_params = {
            **studies['xgb'].best_params,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': 2000,
            'n_jobs': -1,
            'early_stopping_rounds': 100,
            'random_state': seed,
        }
        cat_params = {
            **studies['cat'].best_params,
            'loss_function': 'Logloss',
            'iterations': 2000,
            'verbose': False,
            'thread_count': -1,
            'random_seed': seed,
        }
        if v33.HAS_CUDA:
            lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
            xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
            cat_params.update({'task_type': 'GPU'})
        else:
            lgb_params.update({'device': 'cpu'})

        model_lgb = lgb.LGBMClassifier(**lgb_params)
        try:
            model_lgb.fit(
                x_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
            )
        except Exception:
            lgb_params['device'] = 'cpu'
            model_lgb = lgb.LGBMClassifier(**lgb_params)
            model_lgb.fit(
                x_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
            )

        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=(x_val, y_val),
            early_stopping_rounds=100,
            verbose=False,
        )

        val_preds['lgb'] += model_lgb.predict_proba(x_val)[:, 1] / len(SEEDS)
        val_preds['xgb'] += model_xgb.predict_proba(x_val)[:, 1] / len(SEEDS)
        val_preds['cat'] += model_cat.predict_proba(x_val)[:, 1] / len(SEEDS)
        best_rounds['lgb'].append(int(getattr(model_lgb, 'best_iteration_', 300) or 300))
        best_rounds['xgb'].append(int(getattr(model_xgb, 'best_iteration', 300) or 300))
        best_rounds['cat'].append(int(getattr(model_cat, 'get_best_iteration', lambda: 300)() or 300))

    blended_val = np.mean(
        np.column_stack([val_preds['lgb'], val_preds['xgb'], val_preds['cat']]),
        axis=1,
    )
    return val_preds, best_rounds, blended_val


def fit_full_ensemble(
    x_full,
    y_full,
    x_test,
    sample_weight,
    studies,
    best_rounds,
):
    test_preds = {name: np.zeros(len(x_test)) for name in ['lgb', 'xgb', 'cat']}

    n_lgb = max(50, int(np.median(best_rounds['lgb'])))
    n_xgb = max(50, int(np.median(best_rounds['xgb'])))
    n_cat = max(50, int(np.median(best_rounds['cat'])))

    for seed in SEEDS:
        lgb_params = {
            **studies['lgb'].best_params,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': n_lgb,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': seed,
        }
        xgb_params = {
            **studies['xgb'].best_params,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': n_xgb,
            'n_jobs': -1,
            'random_state': seed,
        }
        cat_params = {
            **studies['cat'].best_params,
            'loss_function': 'Logloss',
            'iterations': n_cat,
            'verbose': False,
            'thread_count': -1,
            'random_seed': seed,
        }
        if v33.HAS_CUDA:
            lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
            xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
            cat_params.update({'task_type': 'GPU'})
        else:
            lgb_params.update({'device': 'cpu'})

        model_lgb = lgb.LGBMClassifier(**lgb_params)
        try:
            model_lgb.fit(x_full, y_full, sample_weight=sample_weight)
        except Exception:
            lgb_params['device'] = 'cpu'
            model_lgb = lgb.LGBMClassifier(**lgb_params)
            model_lgb.fit(x_full, y_full, sample_weight=sample_weight)

        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(x_full, y_full, sample_weight=sample_weight, verbose=False)

        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(x_full, y_full, sample_weight=sample_weight)

        test_preds['lgb'] += model_lgb.predict_proba(x_test)[:, 1] / len(SEEDS)
        test_preds['xgb'] += model_xgb.predict_proba(x_test)[:, 1] / len(SEEDS)
        test_preds['cat'] += model_cat.predict_proba(x_test)[:, 1] / len(SEEDS)

    blended_test = np.mean(
        np.column_stack([test_preds['lgb'], test_preds['xgb'], test_preds['cat']]),
        axis=1,
    )
    return test_preds, blended_test, {'lgb': n_lgb, 'xgb': n_xgb, 'cat': n_cat}


def evaluate(train, pred, mask):
    per_target = {
        target: log_loss(
            train.loc[mask, target].values,
            np.clip(pred.loc[mask, target].values, 1e-7, 1 - 1e-7),
        )
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def describe_vs_anchor(submission, anchor):
    ref = anchor[TARGETS].to_numpy().ravel()
    arr = submission[TARGETS].to_numpy().ravel()
    return {
        'corr_vs_anchor': float(np.corrcoef(ref, arr)[0, 1]),
        'mad_vs_anchor': float(np.mean(np.abs(ref - arr))),
        'max_abs_vs_anchor': float(np.max(np.abs(ref - arr))),
        'means': {target: float(submission[target].mean()) for target in TARGETS},
    }


def build_candidate(keys, anchor, tail_preds, tail_mask, blend_weight):
    out = keys.copy()
    for target in TARGETS:
        out[target] = clip_prob(anchor[target])
        if target not in tail_preds:
            continue
        out.loc[tail_mask, target] = clip_prob(
            (1.0 - blend_weight) * anchor.loc[tail_mask, target]
            + blend_weight * tail_preds[target]
        )
    return out


def save_candidate(
    name,
    train,
    keys,
    anchor_oof,
    anchor_sub,
    proxy_tail_mask,
    actual_tail_mask,
    tail_oof_preds,
    tail_sub_preds,
    blend_weight,
):
    oof = build_candidate(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        anchor_oof,
        tail_oof_preds,
        proxy_tail_mask,
        blend_weight,
    )
    submission = build_candidate(
        keys,
        anchor_sub,
        tail_sub_preds,
        actual_tail_mask,
        blend_weight,
    )
    tail_total, tail_per_target = evaluate(train, oof, proxy_tail_mask)
    full_total, full_per_target = evaluate(train, oof, pd.Series(True, index=train.index))

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    dist = describe_vs_anchor(submission, anchor_sub)
    print(f'\n{name}: full_proxy={full_total:.6f} tail_proxy={tail_total:.6f}')
    print(f'  full_per_target={full_per_target}')
    print(f'  tail_per_target={tail_per_target}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')
    return {
        'name': name,
        'blend_weight': blend_weight,
        'full_proxy': full_total,
        'tail_proxy': tail_total,
        'full_per_target': full_per_target,
        'tail_per_target': tail_per_target,
        'submission': str(sub_path),
        'oof_path': str(oof_path),
        'distribution': dist,
    }


def main():
    ensure_dirs()
    log_path = LOG_DIR / f'run_{EXP_TAG}.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG}...')
    print(f'  tail_targets={TAIL_TARGETS}')
    print(f'  te_windows={TE_WINDOWS}')
    print(f'  seeds={SEEDS}')
    print(f'  optuna_trials={N_OPTUNA_TRIALS}')
    print(f'  recency_alpha={RECENCY_ALPHA}')

    train = load_frame(TRAIN_PATH).reset_index(drop=True)
    sub = load_frame(SUB_PATH).reset_index(drop=True)
    keys = sub[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    anchor_oof = load_frame(ANCHOR_OOF_PATH).reset_index(drop=True)
    anchor_sub = load_frame(ANCHOR_SUB_PATH).reset_index(drop=True)

    proxy_tail_mask, tail_lengths = build_proxy_tail_mask(train, sub)
    actual_tail_mask = build_actual_tail_mask(train, sub)
    print(f'proxy tail rows={int(proxy_tail_mask.sum())} actual tail rows={int(actual_tail_mask.sum())}')
    print(f'proxy tail lengths={tail_lengths}')

    train_full, test_full, feature_cols = v33.build_feature_table(train, sub)
    train_full, test_full, shape_cols = add_visible_history_shape_features(
        train_full,
        test_full,
        proxy_tail_mask,
    )
    feature_cols = feature_cols + shape_cols

    visible_mask = ~proxy_tail_mask
    tail_oof_preds = {}
    tail_sub_preds = {}
    target_summaries = {}

    for target in TAIL_TARGETS:
        print(f'\n{"=" * 72}\nTail target: {target}\n{"=" * 72}')

        visible_hist = train_full.loc[visible_mask, ['subject_id', 'lifelog_date', target]].copy()
        x_visible_base = train_full.loc[visible_mask, ['subject_id', 'lifelog_date'] + feature_cols].copy()
        x_tail_base = train_full.loc[proxy_tail_mask, ['subject_id', 'lifelog_date'] + feature_cols].copy()
        x_full_base = train_full[['subject_id', 'lifelog_date'] + feature_cols].copy()
        x_test_base = test_full[['subject_id', 'lifelog_date'] + feature_cols].copy()

        x_visible, x_tail, _ = add_forward_only_te(x_visible_base, x_tail_base, visible_hist, target)
        x_full, x_test, _ = add_forward_only_te(
            x_full_base,
            x_test_base,
            train_full[['subject_id', 'lifelog_date', target]].copy(),
            target,
        )

        x_visible, x_tail, cross_cols = add_anchor_cross_features(
            x_visible,
            x_tail,
            anchor_oof.loc[visible_mask].reset_index(drop=True),
            anchor_oof.loc[proxy_tail_mask].reset_index(drop=True),
            target,
        )
        x_full, x_test, _ = add_anchor_cross_features(
            x_full,
            x_test,
            anchor_oof,
            anchor_sub,
            target,
        )

        model_cols = [
            col for col in x_visible.columns
            if col not in ['subject_id', 'lifelog_date']
        ]
        y_visible = train_full.loc[visible_mask, target].to_numpy()
        y_tail = train_full.loc[proxy_tail_mask, target].to_numpy()
        stable_cols = select_stable_features(x_visible[model_cols], y_visible)
        print(f'  features: base={len(model_cols)} stable={len(stable_cols)} cross={cross_cols}')

        xv = x_visible[stable_cols].reset_index(drop=True)
        xt = x_tail[stable_cols].reset_index(drop=True)
        xf = x_full[stable_cols].reset_index(drop=True)
        xq = x_test[stable_cols].reset_index(drop=True)

        visible_weights = build_recency_weights(train_full.loc[visible_mask].reset_index(drop=True))
        full_weights = build_recency_weights(train_full.reset_index(drop=True))

        studies = tune_params(xv, y_visible, xt, y_tail, visible_weights)
        print(
            '  tuned tail losses: '
            f"LGB={studies['lgb'].best_value:.6f} "
            f"XGB={studies['xgb'].best_value:.6f} "
            f"CAT={studies['cat'].best_value:.6f}"
        )

        val_preds, best_rounds, blended_val = fit_validation_ensemble(
            xv,
            y_visible,
            xt,
            y_tail,
            visible_weights,
            studies,
        )
        _, blended_test, final_rounds = fit_full_ensemble(
            xf,
            train_full[target].to_numpy(),
            xq,
            full_weights,
            studies,
            best_rounds,
        )

        tail_oof_preds[target] = clip_prob(blended_val)
        tail_sub_preds[target] = clip_prob(blended_test[actual_tail_mask.to_numpy()])

        target_summary = {
            'stable_features': len(stable_cols),
            'tail_losses': {
                'lgb': float(log_loss(y_tail, val_preds['lgb'])),
                'xgb': float(log_loss(y_tail, val_preds['xgb'])),
                'cat': float(log_loss(y_tail, val_preds['cat'])),
                'blend_avg': float(log_loss(y_tail, blended_val)),
                'anchor': float(log_loss(
                    y_tail,
                    anchor_oof.loc[proxy_tail_mask, target].to_numpy(),
                )),
            },
            'best_params': {
                name: studies[name].best_params
                for name in ['lgb', 'xgb', 'cat']
            },
            'final_rounds': final_rounds,
        }
        target_summaries[target] = target_summary
        print(f'  tail_losses={target_summary["tail_losses"]}')
        print(f'  final_rounds={final_rounds}')

    summaries = []
    for name, weight in [
        (f'{EXP_TAG}_w25', 0.25),
        (f'{EXP_TAG}_w50', 0.50),
        (f'{EXP_TAG}_w75', 0.75),
        (f'{EXP_TAG}_full', 1.00),
    ]:
        summaries.append(save_candidate(
            name,
            train,
            keys,
            anchor_oof,
            anchor_sub,
            proxy_tail_mask,
            actual_tail_mask,
            tail_oof_preds,
            tail_sub_preds,
            weight,
        ))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'anchor': str(ANCHOR_SUB_PATH),
        'tail_targets': TAIL_TARGETS,
        'te_windows': TE_WINDOWS,
        'seeds': SEEDS,
        'optuna_trials': N_OPTUNA_TRIALS,
        'recency_alpha': RECENCY_ALPHA,
        'proxy_tail_rows': int(proxy_tail_mask.sum()),
        'actual_tail_rows': int(actual_tail_mask.sum()),
        'proxy_tail_lengths': tail_lengths,
        'target_summaries': target_summaries,
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

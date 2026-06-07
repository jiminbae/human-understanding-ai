# v52: fold-safe target-specialized raw-sensor models for Q1/S1/S2.
#
# Public feedback from v48 consistently says that v47 raw-sensor signal is
# useful mainly for Q1, S1 and S2. This script retrains only those targets with:
#   - relative-date block CV instead of shuffled random CV,
#   - feature selection inside each fold,
#   - target-specific model blending,
#   - seed/model agreement diagnostics,
#   - individual target ablations and combined candidates.
#
# It reuses the cached v47 sensor feature table, so raw parquet extraction is
# not repeated.
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

from jimin.analysis import pseudo_public_interior_profile_eval as profile_eval


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
SPECIAL_TARGETS = ['Q1', 'S1', 'S2']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
LOG_DIR = OUTPUTS_DIR / 'log'
FEATURE_CACHE = OUTPUTS_DIR / 'features' / 'features_v47_hourgrid_subject_state_residual.pkl'

ANCHOR_OOF = OOF_DIR / 'oof_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
ANCHOR_SUB = SUB_DIR / 'submission_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
V47_RAW_OOF = OOF_DIR / 'oof_v47_hourgrid_subject_state_residual_raw.csv'
V47_RAW_SUB = SUB_DIR / 'submission_v47_hourgrid_subject_state_residual_raw.csv'

DEFAULT_BASE_TAGS = [
    # avg310 is the latest public-confirmed best at the time v52 was designed.
    'v48_target_delta_scaled_avg310_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg270_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg230_q2cap115_q3s3guard',
]

# Public-confirmed avg310 weights used when replacing v47 raw with v52 raw.
PUBLIC_REPLACE_WEIGHTS = {
    'Q1': 0.4229500007784415,
    'S1': 0.41969914790546015,
    'S2': 0.3759290712773749,
}

N_FOLDS = int(os.environ.get('V52_N_FOLDS', '5'))
N_SEEDS = int(os.environ.get('V52_N_SEEDS', '3'))
SEED_POOL = [42, 2025, 777, 1234, 314, 9999]
SELECT_ESTIMATORS = int(os.environ.get('V52_SELECT_ESTIMATORS', '420'))
N_ESTIMATORS = int(os.environ.get('V52_N_ESTIMATORS', '1300'))
EXTRA_ESTIMATORS = int(os.environ.get('V52_EXTRA_ESTIMATORS', '480'))
TOP_FEATURES = int(os.environ.get('V52_TOP_FEATURES', '300'))
MAX_POOL_FEATURES = int(os.environ.get('V52_MAX_POOL_FEATURES', '2300'))
FULL_MODEL_BLEND = float(os.environ.get('V52_FULL_MODEL_BLEND', '0.25'))
RUN_FULL_MODELS = os.environ.get('V52_RUN_FULL_MODELS', '1') == '1'

TARGET_TOKENS = {
    'Q1': [
        'slp_', 'sleep', 'night', 'late', 'presleep', 'screen', 'charge',
        'ac_', 'act_', 'light', 'hr_', 'pedo', 'state_', 'subject', 'dow',
        'week', 'month', 'cal_', 'usage', 'amb_',
    ],
    'S1': [
        'slp_', 'sleep', 'night', 'late', 'presleep', 'screen', 'charge',
        'ac_', 'act_', 'light', 'hr_', 'pedo', 'state_', 'subject', 'dow',
        'week', 'month', 'cal_', 'usage', 'amb_',
    ],
    'S2': [
        'slp_', 'sleep', 'night', 'late', 'presleep', 'screen', 'charge',
        'ac_', 'act_', 'light', 'hr_', 'pedo', 'state_', 'subject', 'dow',
        'week', 'month', 'cal_', 'usage', 'amb_', 'ble', 'wifi', 'gps',
    ],
}

# Split-role columns can make proxy OOF look excellent without learning sensor
# signal. Keep subject identity/calendar, but drop direct train/test-position
# encodings from the specialized raw model.
EXCLUDED_FEATURE_TOKENS = [
    'subject_n_rows_all',
    'subject_order',
    'subject_pos_frac',
    'gap_prev_sleep',
    'gap_next_sleep',
]


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
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['sleep_date', 'lifelog_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.reset_index(drop=True)


def choose_base_prediction():
    env_tag = os.environ.get('V52_BASE_TAG')
    tags = [env_tag] if env_tag else []
    tags.extend(DEFAULT_BASE_TAGS)
    for tag in tags:
        if not tag:
            continue
        oof_path = OOF_DIR / f'oof_{tag}.csv'
        sub_path = SUB_DIR / f'submission_{tag}.csv'
        if oof_path.exists() and sub_path.exists():
            return tag, load_frame(oof_path), load_frame(sub_path), str(oof_path), str(sub_path)
    raise FileNotFoundError('No usable v48 base prediction found.')


def load_cached_features():
    if not FEATURE_CACHE.exists():
        raise FileNotFoundError(
            f'Missing v47 feature cache: {FEATURE_CACHE}. Run v47 feature generation first.'
        )
    cached = pd.read_pickle(FEATURE_CACHE)
    train_full = cached['train_full'].reset_index(drop=True)
    test_full = cached['test_full'].reset_index(drop=True)
    feature_cols = list(cached['feature_cols'])
    return train_full, test_full, feature_cols, cached.get('metadata', {})


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def target_logloss(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(y_pred)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate_frame(train, pred, mask=None):
    if mask is None:
        mask = pd.Series(True, index=train.index)
    per_target = {
        target: target_logloss(train.loc[mask, target], pred.loc[mask, target])
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def make_keys(df):
    return df[KEYS].copy()


def build_relative_block_folds(train, n_folds):
    fold_id = pd.Series(-1, index=train.index, dtype=int)
    for _, grp in train.groupby('subject_id', sort=True):
        idx = grp.sort_values('sleep_date').index.to_numpy()
        assignments = np.floor(np.arange(len(idx)) * n_folds / len(idx)).astype(int)
        assignments = np.clip(assignments, 0, n_folds - 1)
        fold_id.loc[idx] = assignments

    folds = []
    for fold in range(n_folds):
        val_idx = np.flatnonzero(fold_id.to_numpy() == fold)
        train_idx = np.flatnonzero(fold_id.to_numpy() != fold)
        folds.append((train_idx, val_idx))
    return folds, fold_id


def build_proxy_role_masks(train, sub):
    profiles = profile_eval.build_profiles(train, sub)
    _, _, interior = profile_eval.build_interior_masks(train, profiles)
    tail = pd.Series(False, index=train.index)
    for subject_id, grp in train.groupby('subject_id', sort=True):
        n_tail = int(profiles[subject_id]['tail_x_run'])
        idx = grp.sort_values('sleep_date').index.to_numpy()
        tail.loc[idx[-min(n_tail, len(idx)):]] = True
    return {'all': pd.Series(True, index=train.index), 'interior': interior, 'tail': tail}


def valid_sensor_feature(feature):
    lower = feature.lower()
    return not any(token in lower for token in EXCLUDED_FEATURE_TOKENS)


def target_feature_pool(train_full, feature_cols, target):
    usable = []
    for col in feature_cols:
        if col not in train_full.columns or not valid_sensor_feature(col):
            continue
        values = pd.to_numeric(train_full[col], errors='coerce')
        coverage = float(values.notna().mean())
        if coverage < 0.10 or values.nunique(dropna=True) <= 1:
            continue
        usable.append((col, coverage))

    tokens = TARGET_TOKENS[target]
    priority = []
    fallback = []
    for col, coverage in usable:
        lower = col.lower()
        item = (coverage, col)
        if any(token in lower for token in tokens):
            priority.append(item)
        else:
            fallback.append(item)

    priority.sort(reverse=True)
    fallback.sort(reverse=True)
    selected = [col for _, col in priority]
    if len(selected) < MAX_POOL_FEATURES:
        selected.extend(col for _, col in fallback[:MAX_POOL_FEATURES - len(selected)])
    return selected[:MAX_POOL_FEATURES]


def fold_select_features(x_train, y_train, feature_pool, target, fold, seed):
    usable = []
    for col in feature_pool:
        values = pd.to_numeric(x_train[col], errors='coerce')
        if values.notna().mean() < 0.08 or values.nunique(dropna=True) <= 1:
            continue
        usable.append(col)
    if len(usable) <= TOP_FEATURES:
        return usable

    selector = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=SELECT_ESTIMATORS,
        learning_rate=0.035,
        num_leaves=9,
        max_depth=4,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.8,
        reg_lambda=5.0,
        random_state=seed + fold * 101,
        n_jobs=-1,
        verbose=-1,
    )
    selector.fit(x_train[usable], y_train)
    importance = pd.Series(selector.feature_importances_, index=usable).sort_values(ascending=False)
    selected = importance.loc[importance > 0].head(TOP_FEATURES).index.tolist()
    if len(selected) < min(80, TOP_FEATURES):
        selected = usable[:TOP_FEATURES]
    print(f'  [v52] {target} fold={fold}: selected {len(selected)} / {len(usable)}')
    return selected


def fit_lgb_pair(x_train, y_train, x_val, y_val, x_test, seed, fold):
    conservative = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=N_ESTIMATORS,
        learning_rate=0.018,
        num_leaves=7,
        max_depth=3,
        min_child_samples=20,
        subsample=0.88,
        colsample_bytree=0.62,
        reg_alpha=1.0,
        reg_lambda=7.0,
        random_state=seed + fold * 17,
        n_jobs=-1,
        verbose=-1,
    )
    expressive = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=N_ESTIMATORS,
        learning_rate=0.018,
        num_leaves=13,
        max_depth=5,
        min_child_samples=14,
        subsample=0.86,
        colsample_bytree=0.68,
        reg_alpha=0.65,
        reg_lambda=4.5,
        random_state=seed + fold * 29 + 3,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(110, verbose=False), lgb.log_evaluation(-1)]
    conservative.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=callbacks)
    expressive.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=callbacks)
    return {
        'cons_val': conservative.predict_proba(x_val)[:, 1],
        'cons_test': conservative.predict_proba(x_test)[:, 1],
        'cons_best_iter': int(conservative.best_iteration_ or N_ESTIMATORS),
        'expr_val': expressive.predict_proba(x_val)[:, 1],
        'expr_test': expressive.predict_proba(x_test)[:, 1],
        'expr_best_iter': int(expressive.best_iteration_ or N_ESTIMATORS),
    }


def fit_extra(x_train, y_train, x_val, x_test, seed, fold):
    imputer = SimpleImputer(strategy='median')
    train_imp = imputer.fit_transform(x_train)
    val_imp = imputer.transform(x_val)
    test_imp = imputer.transform(x_test)
    model = ExtraTreesClassifier(
        n_estimators=EXTRA_ESTIMATORS,
        max_depth=6,
        min_samples_leaf=8,
        max_features=0.62,
        class_weight='balanced',
        random_state=seed + fold * 37,
        n_jobs=-1,
    )
    model.fit(train_imp, y_train)
    return model.predict_proba(val_imp)[:, 1], model.predict_proba(test_imp)[:, 1]


def choose_component_blend(y, oof_components):
    specs = {
        'cons_only': {'cons': 1.00, 'expr': 0.00, 'extra': 0.00},
        'lgb_pair': {'cons': 0.58, 'expr': 0.42, 'extra': 0.00},
        'conservative': {'cons': 0.72, 'expr': 0.20, 'extra': 0.08},
        'balanced': {'cons': 0.55, 'expr': 0.30, 'extra': 0.15},
        'diverse': {'cons': 0.42, 'expr': 0.28, 'extra': 0.30},
    }
    results = {}
    for name, weights in specs.items():
        pred = sum(weights[key] * oof_components[key] for key in weights)
        results[name] = {
            'weights': weights,
            'loss': target_logloss(y, pred),
        }
    best_name = min(results, key=lambda name: results[name]['loss'])
    return best_name, results[best_name]['weights'], results


def train_specialized_target(train_full, test_full, feature_pool, folds, target):
    y = train_full[target].to_numpy(dtype=int)
    n_rows = len(train_full)
    n_test = len(test_full)
    seeds = SEED_POOL[:max(1, min(N_SEEDS, len(SEED_POOL)))]

    oof_components = {
        'cons': np.zeros(n_rows, dtype=float),
        'expr': np.zeros(n_rows, dtype=float),
        'extra': np.zeros(n_rows, dtype=float),
    }
    test_components_cv = {
        'cons': np.zeros(n_test, dtype=float),
        'expr': np.zeros(n_test, dtype=float),
        'extra': np.zeros(n_test, dtype=float),
    }
    oof_member_values = [[] for _ in range(n_rows)]
    test_members = []
    selected_counter = Counter()
    cons_best_iters = []
    expr_best_iters = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        x_train_all = train_full.iloc[train_idx]
        x_val_all = train_full.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        selected = fold_select_features(
            x_train_all,
            y_train,
            feature_pool,
            target,
            fold,
            seeds[0],
        )
        selected_counter.update(selected)
        x_train = x_train_all[selected]
        x_val = x_val_all[selected]
        x_test = test_full[selected]

        fold_components = {
            'cons': np.zeros(len(val_idx), dtype=float),
            'expr': np.zeros(len(val_idx), dtype=float),
            'extra': np.zeros(len(val_idx), dtype=float),
        }
        fold_test = {
            'cons': np.zeros(n_test, dtype=float),
            'expr': np.zeros(n_test, dtype=float),
            'extra': np.zeros(n_test, dtype=float),
        }

        for seed in seeds:
            lgb_pred = fit_lgb_pair(x_train, y_train, x_val, y_val, x_test, seed, fold)
            ext_val, ext_test = fit_extra(x_train, y_train, x_val, x_test, seed, fold)
            fold_components['cons'] += lgb_pred['cons_val'] / len(seeds)
            fold_components['expr'] += lgb_pred['expr_val'] / len(seeds)
            fold_components['extra'] += ext_val / len(seeds)
            fold_test['cons'] += lgb_pred['cons_test'] / len(seeds)
            fold_test['expr'] += lgb_pred['expr_test'] / len(seeds)
            fold_test['extra'] += ext_test / len(seeds)
            cons_best_iters.append(lgb_pred['cons_best_iter'])
            expr_best_iters.append(lgb_pred['expr_best_iter'])
            test_members.extend([
                lgb_pred['cons_test'],
                lgb_pred['expr_test'],
                ext_test,
            ])

        for key in oof_components:
            oof_components[key][val_idx] = fold_components[key]
            test_components_cv[key] += fold_test[key] / len(folds)
        for local_idx, row_idx in enumerate(val_idx):
            oof_member_values[row_idx] = [
                fold_components['cons'][local_idx],
                fold_components['expr'][local_idx],
                fold_components['extra'][local_idx],
            ]

    blend_name, blend_weights, blend_results = choose_component_blend(y, oof_components)
    oof_pred = clip_prob(sum(blend_weights[key] * oof_components[key] for key in blend_weights))
    test_pred_cv = clip_prob(sum(blend_weights[key] * test_components_cv[key] for key in blend_weights))

    full_test_pred = test_pred_cv.copy()
    full_selected = [col for col, _ in selected_counter.most_common(TOP_FEATURES)]
    if RUN_FULL_MODELS and full_selected:
        x_all = train_full[full_selected]
        x_test = test_full[full_selected]
        full_components = {
            'cons': np.zeros(n_test, dtype=float),
            'expr': np.zeros(n_test, dtype=float),
            'extra': np.zeros(n_test, dtype=float),
        }
        cons_iters = max(120, int(np.median(cons_best_iters)))
        expr_iters = max(120, int(np.median(expr_best_iters)))
        for seed in seeds:
            cons = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=cons_iters,
                learning_rate=0.018,
                num_leaves=7,
                max_depth=3,
                min_child_samples=20,
                subsample=0.88,
                colsample_bytree=0.62,
                reg_alpha=1.0,
                reg_lambda=7.0,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            expr = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=expr_iters,
                learning_rate=0.018,
                num_leaves=13,
                max_depth=5,
                min_child_samples=14,
                subsample=0.86,
                colsample_bytree=0.68,
                reg_alpha=0.65,
                reg_lambda=4.5,
                random_state=seed + 3,
                n_jobs=-1,
                verbose=-1,
            )
            cons.fit(x_all, y)
            expr.fit(x_all, y)
            full_components['cons'] += cons.predict_proba(x_test)[:, 1] / len(seeds)
            full_components['expr'] += expr.predict_proba(x_test)[:, 1] / len(seeds)

            imputer = SimpleImputer(strategy='median')
            all_imp = imputer.fit_transform(x_all)
            test_imp = imputer.transform(x_test)
            extra = ExtraTreesClassifier(
                n_estimators=EXTRA_ESTIMATORS,
                max_depth=6,
                min_samples_leaf=8,
                max_features=0.62,
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1,
            )
            extra.fit(all_imp, y)
            full_components['extra'] += extra.predict_proba(test_imp)[:, 1] / len(seeds)

        full_test_pred = clip_prob(
            sum(blend_weights[key] * full_components[key] for key in blend_weights)
        )

    test_pred = clip_prob(
        (1.0 - FULL_MODEL_BLEND) * test_pred_cv
        + FULL_MODEL_BLEND * full_test_pred
    )
    oof_uncertainty = np.array([np.std(values) for values in oof_member_values], dtype=float)
    test_uncertainty = np.std(np.vstack(test_members), axis=0)

    diagnostics = {
        'target': target,
        'feature_pool_count': len(feature_pool),
        'fold_selected_unique_count': len(selected_counter),
        'selected_feature_frequency_top100': selected_counter.most_common(100),
        'full_selected_features_top100': full_selected[:100],
        'component_losses': {
            key: target_logloss(y, pred)
            for key, pred in oof_components.items()
        },
        'blend_candidates': blend_results,
        'chosen_blend': blend_name,
        'chosen_blend_weights': blend_weights,
        'oof_loss': target_logloss(y, oof_pred),
        'uncertainty': {
            'oof_mean': float(oof_uncertainty.mean()),
            'oof_q75': float(np.quantile(oof_uncertainty, 0.75)),
            'test_mean': float(test_uncertainty.mean()),
            'test_q75': float(np.quantile(test_uncertainty, 0.75)),
        },
        'median_best_iterations': {
            'conservative': int(np.median(cons_best_iters)),
            'expressive': int(np.median(expr_best_iters)),
        },
    }
    return oof_pred, test_pred, oof_uncertainty, test_uncertainty, diagnostics


def blend_from_base(base, specialized, weights):
    out = base.copy()
    for target, weight in weights.items():
        if weight <= 0:
            continue
        out[target] = clip_prob((1.0 - weight) * base[target] + weight * specialized[target])
    return out


def replace_from_anchor(base, anchor, specialized, targets):
    out = base.copy()
    for target in targets:
        weight = PUBLIC_REPLACE_WEIGHTS[target]
        out[target] = clip_prob((1.0 - weight) * anchor[target] + weight * specialized[target])
    return out


def agreement_guard_blend(base, anchor, v47_raw, specialized, uncertainty, thresholds, high_weights):
    out = base.copy()
    diagnostics = {}
    for target, high_weight in high_weights.items():
        spec_delta = specialized[target].to_numpy(dtype=float) - anchor[target].to_numpy(dtype=float)
        raw_delta = v47_raw[target].to_numpy(dtype=float) - anchor[target].to_numpy(dtype=float)
        agrees = spec_delta * raw_delta > 0
        stable = uncertainty[target] <= thresholds[target]
        row_weight = np.zeros(len(base), dtype=float)
        row_weight[agrees & stable] = high_weight
        row_weight[agrees & (~stable)] = high_weight * 0.25
        out[target] = clip_prob(
            (1.0 - row_weight) * base[target].to_numpy(dtype=float)
            + row_weight * specialized[target].to_numpy(dtype=float)
        )
        diagnostics[target] = {
            'high_weight': high_weight,
            'uncertainty_threshold': float(thresholds[target]),
            'agreement_rate': float(agrees.mean()),
            'stable_rate': float(stable.mean()),
            'mean_row_weight': float(row_weight.mean()),
        }
    return out, diagnostics


def describe_vs_base(pred, base):
    pred_arr = pred[TARGETS].to_numpy(dtype=float)
    base_arr = base[TARGETS].to_numpy(dtype=float)
    diff = pred_arr - base_arr
    return {
        'corr_vs_base': float(np.corrcoef(pred_arr.ravel(), base_arr.ravel())[0, 1]),
        'mad_vs_base': float(np.mean(np.abs(diff))),
        'max_abs_vs_base': float(np.max(np.abs(diff))),
        'per_target_mad': {
            target: float(np.mean(np.abs(pred[target] - base[target])))
            for target in TARGETS
        },
        'means': {target: float(pred[target].mean()) for target in TARGETS},
    }


def save_candidate(name, train, sub, base_sub, oof_pred, sub_pred, policy, role_masks):
    oof = make_keys(train)
    submission = make_keys(sub)
    for target in TARGETS:
        oof[target] = clip_prob(oof_pred[target])
        submission[target] = clip_prob(sub_pred[target])
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    role_eval = {}
    for role, mask in role_masks.items():
        loss, per_target = evaluate_frame(train, oof, mask)
        role_eval[role] = {'loss': loss, 'per_target': per_target, 'n_rows': int(mask.sum())}
    return {
        'name': name,
        'policy': policy,
        'role_eval': role_eval,
        'distribution_vs_base': describe_vs_base(submission, base_sub),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
    }


def main():
    ensure_dirs()
    log_path = LOG_DIR / 'run_v52_target_specialized_foldsafe_sensor.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_SAMPLE_PATH)
    train_full, test_full, feature_cols, feature_metadata = load_cached_features()
    base_tag, base_oof, base_sub, base_oof_path, base_sub_path = choose_base_prediction()
    anchor_oof = load_frame(ANCHOR_OOF)
    anchor_sub = load_frame(ANCHOR_SUB)
    v47_raw_oof = load_frame(V47_RAW_OOF)
    v47_raw_sub = load_frame(V47_RAW_SUB)
    folds, fold_ids = build_relative_block_folds(train, N_FOLDS)
    role_masks = build_proxy_role_masks(train, sub)

    print('[v52] starting target-specialized fold-safe sensor training')
    print(f'[v52] base={base_tag}')
    print(f'[v52] features={len(feature_cols)} folds={N_FOLDS} seeds={N_SEEDS}')

    specialized_oof = base_oof.copy()
    specialized_sub = base_sub.copy()
    oof_uncertainty = {}
    test_uncertainty = {}
    target_diagnostics = {}
    feature_pools = {}

    for target in SPECIAL_TARGETS:
        print(f'\n[v52] target={target}')
        pool = target_feature_pool(train_full, feature_cols, target)
        feature_pools[target] = pool
        print(f'[v52] {target}: target feature pool={len(pool)}')
        oof_pred, sub_pred, oof_unc, test_unc, diagnostics = train_specialized_target(
            train_full,
            test_full,
            pool,
            folds,
            target,
        )
        specialized_oof[target] = oof_pred
        specialized_sub[target] = sub_pred
        oof_uncertainty[target] = oof_unc
        test_uncertainty[target] = test_unc
        target_diagnostics[target] = diagnostics
        print(f'[v52] {target}: specialized block-oof={diagnostics["oof_loss"]:.6f}')

    raw_name = 'v52_target_specialized_foldsafe_sensor_raw'
    raw_oof_path = OOF_DIR / f'oof_{raw_name}.csv'
    raw_sub_path = SUB_DIR / f'submission_{raw_name}.csv'
    specialized_oof.to_csv(raw_oof_path, index=False)
    specialized_sub.to_csv(raw_sub_path, index=False)

    candidates = [{
        'name': raw_name,
        'policy': {'type': 'specialized_raw_q1s1s2_base_others'},
        'role_eval': {
            role: {
                'loss': evaluate_frame(train, specialized_oof, mask)[0],
                'per_target': evaluate_frame(train, specialized_oof, mask)[1],
                'n_rows': int(mask.sum()),
            }
            for role, mask in role_masks.items()
        },
        'distribution_vs_base': describe_vs_base(specialized_sub, base_sub),
        'oof_path': str(raw_oof_path),
        'submission': str(raw_sub_path),
    }]

    for target in SPECIAL_TARGETS:
        oof_pred = replace_from_anchor(base_oof, anchor_oof, specialized_oof, [target])
        sub_pred = replace_from_anchor(base_sub, anchor_sub, specialized_sub, [target])
        candidates.append(save_candidate(
            f'v52_{target.lower()}_specialized_replace_v47',
            train,
            sub,
            base_sub,
            oof_pred,
            sub_pred,
            {
                'type': 'individual_target_replace_v47_raw',
                'target': target,
                'replace_weight': PUBLIC_REPLACE_WEIGHTS[target],
            },
            role_masks,
        ))

    oof_pred = replace_from_anchor(base_oof, anchor_oof, specialized_oof, SPECIAL_TARGETS)
    sub_pred = replace_from_anchor(base_sub, anchor_sub, specialized_sub, SPECIAL_TARGETS)
    candidates.append(save_candidate(
        'v52_q1s1s2_specialized_replace_v47_combined',
        train,
        sub,
        base_sub,
        oof_pred,
        sub_pred,
        {
            'type': 'combined_replace_v47_raw',
            'replace_weights': PUBLIC_REPLACE_WEIGHTS,
        },
        role_masks,
    ))

    add_specs = {
        'v52_q1s1s2_specialized_add_safe': {'Q1': 0.06, 'S1': 0.06, 'S2': 0.06},
        'v52_q1s1s2_specialized_add_mid': {'Q1': 0.12, 'S1': 0.12, 'S2': 0.12},
        'v52_q1s1s2_specialized_add_targeted': {'Q1': 0.14, 'S1': 0.10, 'S2': 0.10},
    }
    for name, weights in add_specs.items():
        oof_pred = blend_from_base(base_oof, specialized_oof, weights)
        sub_pred = blend_from_base(base_sub, specialized_sub, weights)
        candidates.append(save_candidate(
            name,
            train,
            sub,
            base_sub,
            oof_pred,
            sub_pred,
            {'type': 'base_add_specialized', 'weights': weights},
            role_masks,
        ))

    thresholds = {
        target: float(np.quantile(oof_uncertainty[target], 0.75))
        for target in SPECIAL_TARGETS
    }
    guard_weights = {'Q1': 0.18, 'S1': 0.16, 'S2': 0.16}
    oof_guard, oof_guard_diag = agreement_guard_blend(
        base_oof,
        anchor_oof,
        v47_raw_oof,
        specialized_oof,
        oof_uncertainty,
        thresholds,
        guard_weights,
    )
    sub_guard, sub_guard_diag = agreement_guard_blend(
        base_sub,
        anchor_sub,
        v47_raw_sub,
        specialized_sub,
        test_uncertainty,
        thresholds,
        guard_weights,
    )
    candidates.append(save_candidate(
        'v52_q1s1s2_specialized_agreement_guarded',
        train,
        sub,
        base_sub,
        oof_guard,
        sub_guard,
        {
            'type': 'v47_v52_direction_agreement_and_uncertainty_guard',
            'high_weights': guard_weights,
            'oof_diagnostics': oof_guard_diag,
            'submission_diagnostics': sub_guard_diag,
        },
        role_masks,
    ))

    candidates = sorted(candidates, key=lambda item: item['role_eval']['all']['loss'])
    baseline_role_eval = {}
    for source_name, source in [('base', base_oof), ('v47_raw', v47_raw_oof)]:
        baseline_role_eval[source_name] = {}
        for role, mask in role_masks.items():
            loss, per_target = evaluate_frame(train, source, mask)
            baseline_role_eval[source_name][role] = {
                'loss': loss,
                'per_target': per_target,
                'n_rows': int(mask.sum()),
            }

    summary = {
        'exp_tag': 'v52_target_specialized_foldsafe_sensor',
        'config': {
            'n_folds': N_FOLDS,
            'n_seeds': N_SEEDS,
            'seeds': SEED_POOL[:max(1, min(N_SEEDS, len(SEED_POOL)))],
            'select_estimators': SELECT_ESTIMATORS,
            'n_estimators': N_ESTIMATORS,
            'extra_estimators': EXTRA_ESTIMATORS,
            'top_features': TOP_FEATURES,
            'max_pool_features': MAX_POOL_FEATURES,
            'run_full_models': RUN_FULL_MODELS,
            'full_model_blend': FULL_MODEL_BLEND,
            'excluded_feature_tokens': EXCLUDED_FEATURE_TOKENS,
        },
        'base': {
            'tag': base_tag,
            'oof_path': base_oof_path,
            'submission_path': base_sub_path,
        },
        'feature_cache': str(FEATURE_CACHE),
        'feature_metadata': feature_metadata,
        'fold_rows': {
            str(fold): int((fold_ids == fold).sum())
            for fold in range(N_FOLDS)
        },
        'role_rows': {
            role: int(mask.sum())
            for role, mask in role_masks.items()
        },
        'baseline_role_eval': baseline_role_eval,
        'target_diagnostics': target_diagnostics,
        'feature_pool_counts': {
            target: len(pool)
            for target, pool in feature_pools.items()
        },
        'candidates': candidates,
        'notes': [
            'All feature selection is performed inside each relative-date block fold.',
            'Direct split-position features are excluded from specialized raw models.',
            'Individual replace candidates are the cleanest public ablations for Q1/S1/S2.',
            'Block OOF and historical base OOF are not perfectly comparable; public ablations remain decisive.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v52_target_specialized_foldsafe_sensor.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('\n[v52] top candidates by all-row block OOF:')
    for item in candidates:
        print(
            f'  {item["name"]}: '
            f'all={item["role_eval"]["all"]["loss"]:.6f} '
            f'interior={item["role_eval"]["interior"]["loss"]:.6f} '
            f'tail={item["role_eval"]["tail"]["loss"]:.6f}'
        )
    print(f'[v52] summary={summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

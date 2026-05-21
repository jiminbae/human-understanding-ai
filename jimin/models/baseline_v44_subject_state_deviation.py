# v44: subject-state deviation base predictor.
#   - v38/w40 is the public-best anchor; recent v39-v43 post-processing probes
#     suggest that reuse/blend surgery is nearly saturated.
#   - This script builds a deliberately different base model from sensor values
#     expressed relative to each subject's usual state.
#   - Submissions are conservative blends with v38/w40, not raw replacements.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jimin.analysis import pseudo_public_interior_profile_eval as interior_eval
from jimin.models import baseline_v33_long_history_cross_target as v33
from jimin.models import baseline_v38_block_role_aware as v38


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
EXP_TAG = os.environ.get('V44_EXP_TAG', 'v44_subject_state_deviation')
MAX_STATE_COLS = int(os.environ.get('V44_MAX_STATE_COLS', '90'))
MAX_MODEL_FEATURES = int(os.environ.get('V44_MAX_MODEL_FEATURES', '95'))
SEEDS = [int(x) for x in os.environ.get('V44_SEEDS', '42,2025').split(',') if x.strip()]
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.25]

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_OOF_PATH = OOF_DIR / 'oof_v38_block_role_aware_tail_conservative_w40.csv'
ANCHOR_SUB_PATH = SUB_DIR / 'submission_v38_block_role_aware_tail_conservative_w40.csv'


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
    return df.reset_index(drop=True)


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def build_actual_interior_mask(train, sub):
    mask = np.zeros(len(sub), dtype=bool)
    for sid, grp in sub.groupby('subject_id', sort=True):
        train_dates = train.loc[train['subject_id'] == sid, 'sleep_date']
        for idx, sleep_date in grp['sleep_date'].items():
            mask[idx] = bool((train_dates > sleep_date).any())
    return pd.Series(mask, index=sub.index)


def choose_state_columns(train_full, test_full, feature_cols):
    combined = pd.concat([
        train_full[['subject_id'] + feature_cols],
        test_full[['subject_id'] + feature_cols],
    ], axis=0, ignore_index=True)

    sensor_prefixes = (
        'act_', 'pedo_', 'hr_', 'screen_', 'mlight_', 'wlight_', 'ac_',
        'gps_', 'usage_', 'wifi_', 'ble_', 'amb_', 'slp_',
    )
    excluded = {
        'subject_num', 'dow', 'month', 'week', 'is_weekend',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    }
    candidates = []
    for col in feature_cols:
        if col in excluded or col.endswith('_subj_z'):
            continue
        if not col.startswith(sensor_prefixes):
            continue
        values = pd.to_numeric(combined[col], errors='coerce')
        nonnull = float(values.notna().mean())
        if nonnull < 0.35:
            continue
        var = float(values.var(skipna=True))
        if not np.isfinite(var) or var <= 1e-12:
            continue
        score = nonnull * np.log1p(var)
        candidates.append((score, col))

    candidates = sorted(candidates, reverse=True)
    selected = [col for _, col in candidates[:MAX_STATE_COLS]]
    print(f'[v44] selected state columns: {len(selected)} / {len(candidates)}')
    return selected


def add_subject_state_features(train_full, test_full, feature_cols):
    selected = choose_state_columns(train_full, test_full, feature_cols)
    combined = pd.concat([
        train_full[['subject_id'] + selected].assign(_split='train'),
        test_full[['subject_id'] + selected].assign(_split='test'),
    ], axis=0, ignore_index=True)

    new_cols = []
    z_cols = []
    for col in selected:
        values = pd.to_numeric(combined[col], errors='coerce')
        grouped = values.groupby(combined['subject_id'])
        mu = grouped.transform('mean')
        sig = grouped.transform('std').replace(0, np.nan)
        median = grouped.transform('median')

        z_col = f'v44_{col}_subj_z'
        abs_col = f'v44_{col}_subj_absz'
        med_col = f'v44_{col}_subj_med_delta'
        pct_col = f'v44_{col}_subj_pct'
        combined[z_col] = ((values - mu) / sig).clip(-6, 6)
        combined[abs_col] = combined[z_col].abs()
        combined[med_col] = values - median
        combined[pct_col] = values.groupby(combined['subject_id']).rank(pct=True)
        z_cols.append(z_col)
        new_cols.extend([z_col, abs_col, med_col, pct_col])

    if z_cols:
        combined['v44_state_absz_mean'] = combined[z_cols].abs().mean(axis=1)
        combined['v44_state_absz_max'] = combined[z_cols].abs().max(axis=1)
        combined['v44_state_missing_frac'] = combined[selected].isna().mean(axis=1)
        new_cols.extend(['v44_state_absz_mean', 'v44_state_absz_max', 'v44_state_missing_frac'])

    time_cols = []
    for col in ['dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'is_weekend']:
        if col not in train_full.columns:
            continue
        new_col = f'v44_time_{col}'
        combined[new_col] = pd.concat([train_full[col], test_full[col]], ignore_index=True)
        time_cols.append(new_col)

    train_state = combined.loc[combined['_split'] == 'train', new_cols + time_cols].reset_index(drop=True)
    test_state = combined.loc[combined['_split'] == 'test', new_cols + time_cols].reset_index(drop=True)
    state_cols = train_state.columns.tolist()
    print(f'[v44] state feature count: {len(state_cols)}')

    train_aug = pd.concat([train_full.reset_index(drop=True), train_state], axis=1)
    test_aug = pd.concat([test_full.reset_index(drop=True), test_state], axis=1)
    return train_aug, test_aug, state_cols, selected


def select_model_features(x, y):
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=250,
        learning_rate=0.035,
        num_leaves=7,
        min_child_samples=18,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_alpha=0.4,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    x_imp = x.copy()
    med = x_imp.median(numeric_only=True)
    x_imp = x_imp.fillna(med)
    model.fit(x_imp, y)
    imp = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    selected = imp.loc[imp > 0].index.tolist()[:MAX_MODEL_FEATURES]
    if not selected:
        selected = x.columns.tolist()[:MAX_MODEL_FEATURES]
    return selected


def fit_predict_target(x_train, y, x_test, target):
    class_counts = np.bincount(y.astype(int), minlength=2)
    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        mean_prob = float(np.clip(y.mean(), 0.02, 0.98))
        return (
            np.full(len(x_train), mean_prob),
            np.full(len(x_test), mean_prob),
            [],
        )

    selected = select_model_features(x_train, y)
    x_train = x_train[selected]
    x_test = x_test[selected]

    oof_lgb = np.zeros(len(x_train), dtype=float)
    oof_lr = np.zeros(len(x_train), dtype=float)
    test_lgb = np.zeros(len(x_test), dtype=float)
    test_lr = np.zeros(len(x_test), dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y), start=1):
        x_tr = x_train.iloc[tr_idx]
        x_val = x_train.iloc[val_idx]
        y_tr = y[tr_idx]

        fold_lgb_test = np.zeros(len(x_test), dtype=float)
        fold_lgb_val = np.zeros(len(val_idx), dtype=float)
        for seed in SEEDS:
            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=320,
                learning_rate=0.025,
                num_leaves=7,
                min_child_samples=16,
                subsample=0.85,
                colsample_bytree=0.70,
                reg_alpha=0.5,
                reg_lambda=2.5,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(x_tr, y_tr)
            fold_lgb_val += lgb_model.predict_proba(x_val)[:, 1] / len(SEEDS)
            fold_lgb_test += lgb_model.predict_proba(x_test)[:, 1] / len(SEEDS)
        oof_lgb[val_idx] = fold_lgb_val
        test_lgb += fold_lgb_test / n_splits

        lr = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=0.18, max_iter=2000, solver='lbfgs')),
        ])
        lr.fit(x_tr, y_tr)
        oof_lr[val_idx] = lr.predict_proba(x_val)[:, 1]
        test_lr += lr.predict_proba(x_test)[:, 1] / n_splits

    oof_avg = clip_prob(0.65 * oof_lgb + 0.35 * oof_lr)
    test_avg = clip_prob(0.65 * test_lgb + 0.35 * test_lr)
    print(
        f'  {target}: features={len(selected)} '
        f'LGB={log_loss(y, clip_prob(oof_lgb)):.6f} '
        f'LR={log_loss(y, clip_prob(oof_lr)):.6f} '
        f'AVG={log_loss(y, oof_avg):.6f}'
    )
    return oof_avg, test_avg, selected


def train_state_model(train_full, test_full, state_cols):
    oof = pd.DataFrame(index=train_full.index)
    sub = pd.DataFrame(index=test_full.index)
    selected_by_target = {}
    x_train = train_full[state_cols].copy()
    x_test = test_full[state_cols].copy()
    for target in TARGETS:
        y = train_full[target].astype(int).to_numpy()
        oof[target], sub[target], selected = fit_predict_target(x_train, y, x_test, target)
        selected_by_target[target] = selected
    return oof, sub, selected_by_target


def evaluate(train, pred, mask):
    per_target = {
        target: float(log_loss(
            train.loc[mask, target].values,
            np.clip(pred.loc[mask, target].values, 1e-7, 1 - 1e-7),
        ))
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


def build_blend(keys, anchor, state_pred, mask, weight_by_target):
    out = keys.copy()
    for target in TARGETS:
        out[target] = clip_prob(anchor[target])
        weight = float(weight_by_target.get(target, 0.0))
        if weight <= 0:
            continue
        out.loc[mask, target] = clip_prob(
            (1.0 - weight) * anchor.loc[mask, target].to_numpy()
            + weight * state_pred.loc[mask, target].to_numpy()
        )
    return out


def save_candidate(
    name,
    train,
    keys,
    anchor_oof,
    anchor_sub,
    state_oof,
    state_sub,
    train_mask,
    sub_mask,
    eval_masks,
    weight_by_target,
):
    oof = build_blend(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        anchor_oof,
        state_oof,
        train_mask,
        weight_by_target,
    )
    submission = build_blend(keys, anchor_sub, state_sub, sub_mask, weight_by_target)

    metrics = {}
    for mask_name, mask in eval_masks.items():
        total, per_target = evaluate(train, oof, mask)
        metrics[mask_name] = {'loss': total, 'per_target': per_target}

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    dist = describe_vs_anchor(submission, anchor_sub)
    print(f'\n{name}')
    for mask_name, result in metrics.items():
        print(f'  {mask_name}: {result["loss"]:.6f}')
    print(f'  weights={weight_by_target}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')
    return {
        'name': name,
        'weights': weight_by_target,
        'metrics': metrics,
        'submission': str(sub_path),
        'oof_path': str(oof_path),
        'distribution': dist,
    }


def choose_target_select_weights(anchor_oof, state_oof, train, simple_mask, fragmented_mask, all_mask, base_weight):
    selected = {}
    tmp_all = build_blend(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        anchor_oof,
        state_oof,
        all_mask,
        {target: base_weight for target in TARGETS},
    )
    for target in TARGETS:
        base_all = log_loss(train.loc[all_mask, target], np.clip(anchor_oof.loc[all_mask, target], 1e-7, 1 - 1e-7))
        cand_all = log_loss(train.loc[all_mask, target], np.clip(tmp_all.loc[all_mask, target], 1e-7, 1 - 1e-7))
        base_simple = log_loss(train.loc[simple_mask, target], np.clip(anchor_oof.loc[simple_mask, target], 1e-7, 1 - 1e-7))
        cand_simple = log_loss(train.loc[simple_mask, target], np.clip(tmp_all.loc[simple_mask, target], 1e-7, 1 - 1e-7))
        base_frag = log_loss(train.loc[fragmented_mask, target], np.clip(anchor_oof.loc[fragmented_mask, target], 1e-7, 1 - 1e-7))
        cand_frag = log_loss(train.loc[fragmented_mask, target], np.clip(tmp_all.loc[fragmented_mask, target], 1e-7, 1 - 1e-7))
        if cand_all < base_all and cand_simple <= base_simple + 0.0015 and cand_frag <= base_frag + 0.0015:
            selected[target] = base_weight
    return selected


def main():
    ensure_dirs()
    log_path = LOG_DIR / f'run_{EXP_TAG}.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG}...')
    print(f'  MAX_STATE_COLS={MAX_STATE_COLS} MAX_MODEL_FEATURES={MAX_MODEL_FEATURES} SEEDS={SEEDS}')
    train_df = load_frame(TRAIN_PATH)
    sub_df = load_frame(SUB_PATH)
    keys = sub_df[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    anchor_oof = load_frame(ANCHOR_OOF_PATH)
    anchor_sub = load_frame(ANCHOR_SUB_PATH)

    train_full, test_full, feature_cols = v33.build_feature_table(train_df, sub_df)
    train_full, test_full, state_cols, selected_state_cols = add_subject_state_features(
        train_full, test_full, feature_cols)

    print('\n[v44] training subject-state model...')
    state_oof_targets, state_sub_targets, selected_by_target = train_state_model(
        train_full, test_full, state_cols)
    state_oof = pd.concat([
        train_full[['subject_id', 'sleep_date', 'lifelog_date']].reset_index(drop=True),
        state_oof_targets[TARGETS].reset_index(drop=True),
    ], axis=1)
    state_sub = pd.concat([
        keys.reset_index(drop=True),
        state_sub_targets[TARGETS].reset_index(drop=True),
    ], axis=1)

    state_oof_path = OOF_DIR / f'oof_{EXP_TAG}_raw.csv'
    state_sub_path = SUB_DIR / f'submission_{EXP_TAG}_raw.csv'
    state_oof.to_csv(state_oof_path, index=False)
    state_sub.to_csv(state_sub_path, index=False)

    profiles = interior_eval.build_profiles(train_df, sub_df)
    simple_mask, fragmented_mask, all_interior_mask = interior_eval.build_interior_masks(train_df, profiles)
    old_middle_mask, tail_mask = v38.build_proxy_masks(train_df, sub_df)
    hybrid_mask = all_interior_mask | tail_mask
    actual_interior_mask = build_actual_interior_mask(train_df, sub_df)
    all_train_mask = pd.Series(True, index=train_df.index)
    all_sub_mask = pd.Series(True, index=sub_df.index)

    eval_masks = {
        'all_train': all_train_mask,
        'all_interior': all_interior_mask,
        'simple_interior': simple_mask,
        'fragmented_interior': fragmented_mask,
        'tail': tail_mask,
        'role_hybrid': hybrid_mask,
    }

    anchor_metrics = {
        name: {'loss': evaluate(train_df, anchor_oof, mask)[0]}
        for name, mask in eval_masks.items()
    }
    state_metrics = {
        name: {'loss': evaluate(train_df, state_oof, mask)[0]}
        for name, mask in eval_masks.items()
    }
    print('\n[v44] anchor vs raw state model')
    for name in eval_masks:
        print(f'  {name}: anchor={anchor_metrics[name]["loss"]:.6f} state={state_metrics[name]["loss"]:.6f}')

    summaries = []
    for weight in BLEND_WEIGHTS:
        weight_map = {target: weight for target in TARGETS}
        wtag = f'w{int(round(weight * 100)):02d}'
        summaries.append(save_candidate(
            f'{EXP_TAG}_allrows_{wtag}',
            train_df,
            keys,
            anchor_oof,
            anchor_sub,
            state_oof,
            state_sub,
            all_train_mask,
            all_sub_mask,
            eval_masks,
            weight_map,
        ))
        summaries.append(save_candidate(
            f'{EXP_TAG}_interior_{wtag}',
            train_df,
            keys,
            anchor_oof,
            anchor_sub,
            state_oof,
            state_sub,
            all_interior_mask,
            actual_interior_mask,
            eval_masks,
            weight_map,
        ))

    target_weights = choose_target_select_weights(
        anchor_oof,
        state_oof,
        train_df,
        simple_mask,
        fragmented_mask,
        all_interior_mask,
        0.10,
    )
    if target_weights:
        summaries.append(save_candidate(
            f'{EXP_TAG}_interior_targetselect_w10',
            train_df,
            keys,
            anchor_oof,
            anchor_sub,
            state_oof,
            state_sub,
            all_interior_mask,
            actual_interior_mask,
            eval_masks,
            target_weights,
        ))
    else:
        print('[v44] no target passed target-select gate at w10')

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'anchor': str(ANCHOR_SUB_PATH),
        'raw_state_oof': str(state_oof_path),
        'raw_state_submission': str(state_sub_path),
        'selected_state_cols': selected_state_cols,
        'state_feature_count': len(state_cols),
        'selected_by_target': selected_by_target,
        'anchor_metrics': anchor_metrics,
        'state_metrics': state_metrics,
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

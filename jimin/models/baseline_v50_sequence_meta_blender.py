# v50: sequence-aware meta blender.
#
# v49 showed that direct label propagation is too crude. v50 keeps the same
# sequence information, but uses it as meta features for deciding when to trust
# the current v48 base, v47 raw-sensor prediction, or v49 graph prediction.
#
# The model is intentionally small: target-wise logistic regression + shallow
# ExtraTrees on pseudo-hidden context features. It writes raw meta predictions
# and conservative base/meta blends.
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']
PSEUDO_BLOCK_LENGTHS = [1, 2, 3, 5, 8]
SEEDS = [42, 2025, 777]

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

RAW_OOF = OOF_DIR / 'oof_v47_hourgrid_subject_state_residual_raw.csv'
RAW_SUB = SUB_DIR / 'submission_v47_hourgrid_subject_state_residual_raw.csv'
GRAPH_OOF = OOF_DIR / 'oof_v49_sequence_graph_only.csv'
GRAPH_SUB = SUB_DIR / 'submission_v49_sequence_graph_only.csv'
GRAPH_CONF_OOF = OOF_DIR / 'oof_v49_sequence_graph_confidence.csv'
GRAPH_CONF_SUB = SUB_DIR / 'submission_v49_sequence_graph_confidence.csv'

DEFAULT_BASE_TAGS = [
    'v48_target_delta_scaled_avg310_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg270_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg250_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg230_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg190_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg145_q3s3guard',
]


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['sleep_date', 'lifelog_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.reset_index(drop=True)


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def logit(values):
    p = clip_prob(values)
    return np.log(p / (1.0 - p))


def target_logloss(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = clip_prob(y_pred)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate_frame(train, pred):
    per_target = {
        target: target_logloss(train[target], pred[target])
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def make_keys(df):
    return df[KEYS].copy()


def choose_base_prediction():
    env_tag = os.environ.get('V50_BASE_TAG')
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


def subject_target_priors(train):
    global_mean = {target: float(train[target].mean()) for target in TARGETS}
    subject_mean = {
        target: train.groupby('subject_id')[target].mean().to_dict()
        for target in TARGETS
    }
    return global_mean, subject_mean


def smoothed_subject_mean(subject_id, target, global_mean, subject_mean):
    subj = float(subject_mean[target].get(subject_id, global_mean[target]))
    return 0.78 * subj + 0.22 * global_mean[target]


def prepare_sequence_frame(train, sub=None):
    train_seq = train[KEYS + TARGETS].copy()
    train_seq['_split'] = 'train'
    train_seq['_orig_index'] = np.arange(len(train_seq))
    if sub is None:
        seq = train_seq
    else:
        sub_seq = sub[KEYS].copy()
        for target in TARGETS:
            sub_seq[target] = np.nan
        sub_seq['_split'] = 'test'
        sub_seq['_orig_index'] = np.arange(len(sub_seq))
        seq = pd.concat([train_seq, sub_seq], ignore_index=True)
    return seq.sort_values(['subject_id', 'sleep_date', 'lifelog_date', '_split']).reset_index(drop=True)


def neighbor_positions(known_positions, pos):
    prev_positions = known_positions[known_positions < pos]
    next_positions = known_positions[known_positions > pos]
    prev_pos = int(prev_positions[-1]) if len(prev_positions) else None
    next_pos = int(next_positions[0]) if len(next_positions) else None
    return prev_pos, next_pos


def recent_mean(seq, known_positions, target, pos, side, max_count=3):
    if side == 'prev':
        positions = known_positions[known_positions < pos][-max_count:]
    else:
        positions = known_positions[known_positions > pos][:max_count]
    if len(positions) == 0:
        return np.nan
    return float(seq.loc[positions, target].mean())


def context_for_position(seq, known_positions, pos, target, global_mean, subject_mean):
    row = seq.loc[pos]
    subject_id = row['subject_id']
    prior = smoothed_subject_mean(subject_id, target, global_mean, subject_mean)
    prev_pos, next_pos = neighbor_positions(known_positions, pos)

    prev_label = np.nan
    next_label = np.nan
    dist_prev = np.nan
    dist_next = np.nan
    pos_frac = np.nan

    if prev_pos is not None:
        prev_label = float(seq.loc[prev_pos, target])
        dist_prev = max(1.0, float((row['sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days))
    if next_pos is not None:
        next_label = float(seq.loc[next_pos, target])
        dist_next = max(1.0, float((seq.loc[next_pos, 'sleep_date'] - row['sleep_date']).days))
    if prev_pos is not None and next_pos is not None:
        span = max(1.0, float((seq.loc[next_pos, 'sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days))
        pos_frac = max(0.0, min(1.0, float((row['sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days) / span))

    values = {
        'ctx_prior': prior,
        'ctx_has_prev': float(prev_pos is not None),
        'ctx_has_next': float(next_pos is not None),
        'ctx_is_interior': float(prev_pos is not None and next_pos is not None),
        'ctx_prev_label': prev_label,
        'ctx_next_label': next_label,
        'ctx_prev_next_mean': np.nanmean([prev_label, next_label]),
        'ctx_prev_next_absdiff': abs(prev_label - next_label) if np.isfinite(prev_label) and np.isfinite(next_label) else np.nan,
        'ctx_prev_next_agree': float(np.isfinite(prev_label) and np.isfinite(next_label) and abs(prev_label - next_label) < 1e-12),
        'ctx_dist_prev': dist_prev,
        'ctx_dist_next': dist_next,
        'ctx_dist_min': np.nanmin([dist_prev, dist_next]),
        'ctx_pos_frac': pos_frac,
        'ctx_prev_recent3': recent_mean(seq, known_positions, target, pos, 'prev'),
        'ctx_next_recent3': recent_mean(seq, known_positions, target, pos, 'next'),
    }
    # Avoid all-nan warnings becoming features; imputer will handle partial nan.
    for key, value in list(values.items()):
        if isinstance(value, float) and not np.isfinite(value):
            values[key] = np.nan
    return values


def empty_context_accumulator(n_rows):
    cols = [
        'ctx_prior', 'ctx_has_prev', 'ctx_has_next', 'ctx_is_interior',
        'ctx_prev_label', 'ctx_next_label', 'ctx_prev_next_mean',
        'ctx_prev_next_absdiff', 'ctx_prev_next_agree',
        'ctx_dist_prev', 'ctx_dist_next', 'ctx_dist_min', 'ctx_pos_frac',
        'ctx_prev_recent3', 'ctx_next_recent3',
    ]
    sums = pd.DataFrame(0.0, index=np.arange(n_rows), columns=cols)
    counts = pd.Series(0, index=np.arange(n_rows), dtype=int)
    return sums, counts, cols


def add_context_record(sums, counts, row_idx, record):
    for col, value in record.items():
        if value is not None and np.isfinite(value):
            sums.loc[row_idx, col] += float(value)
    counts.loc[row_idx] += 1


def build_train_context_features(train):
    global_mean, subject_mean = subject_target_priors(train)
    seq = prepare_sequence_frame(train)
    context_by_target = {}
    for target in TARGETS:
        sums, counts, cols = empty_context_accumulator(len(train))
        for subject_id, grp in seq.groupby('subject_id', sort=False):
            grp = grp.reset_index(drop=False).rename(columns={'index': '_seq_index'})
            n = len(grp)
            for length in PSEUDO_BLOCK_LENGTHS:
                if length > n:
                    continue
                for start in range(0, n - length + 1):
                    hidden_positions = set(range(start, start + length))
                    known_positions = np.array([pos for pos in range(n) if pos not in hidden_positions], dtype=int)
                    if len(known_positions) == 0:
                        continue
                    for pos in hidden_positions:
                        row_idx = int(grp.loc[pos, '_orig_index'])
                        rec = context_for_position(grp, known_positions, pos, target, global_mean, subject_mean)
                        add_context_record(sums, counts, row_idx, rec)
        denom = counts.replace(0, np.nan)
        features = sums.div(denom, axis=0)
        for col in cols:
            if col.startswith('ctx_has') or col in ['ctx_is_interior', 'ctx_prev_next_agree']:
                features[col] = features[col].fillna(0.0)
        context_by_target[target] = features.reset_index(drop=True)
    return context_by_target


def build_test_context_features(train, sub):
    global_mean, subject_mean = subject_target_priors(train)
    seq = prepare_sequence_frame(train, sub)
    context_by_target = {
        target: pd.DataFrame(index=np.arange(len(sub)))
        for target in TARGETS
    }
    for target in TARGETS:
        rows = [None] * len(sub)
        for subject_id, grp in seq.groupby('subject_id', sort=False):
            grp = grp.reset_index(drop=True)
            known_positions = grp.index[grp['_split'] == 'train'].to_numpy(dtype=int)
            for pos, row in grp.loc[grp['_split'] == 'test'].iterrows():
                out_idx = int(row['_orig_index'])
                rows[out_idx] = context_for_position(grp, known_positions, int(pos), target, global_mean, subject_mean)
        context_by_target[target] = pd.DataFrame(rows)
    return context_by_target


def calendar_features(frame):
    out = pd.DataFrame(index=frame.index)
    out['subject_num'] = frame['subject_id'].str.extract(r'(\d+)').astype(float)
    out['sleep_dow'] = frame['sleep_date'].dt.dayofweek.astype(float)
    out['sleep_is_weekend'] = (out['sleep_dow'] >= 5).astype(float)
    out['sleep_dow_sin'] = np.sin(2 * np.pi * out['sleep_dow'] / 7)
    out['sleep_dow_cos'] = np.cos(2 * np.pi * out['sleep_dow'] / 7)
    out['lifelog_dow'] = frame['lifelog_date'].dt.dayofweek.astype(float)
    out['lifelog_dow_sin'] = np.sin(2 * np.pi * out['lifelog_dow'] / 7)
    out['lifelog_dow_cos'] = np.cos(2 * np.pi * out['lifelog_dow'] / 7)
    sorted_frame = frame.sort_values(['subject_id', 'sleep_date']).copy()
    sorted_frame['_order'] = sorted_frame.groupby('subject_id').cumcount()
    sorted_frame['_n'] = sorted_frame.groupby('subject_id')['subject_id'].transform('size')
    pos_frac = sorted_frame['_order'] / (sorted_frame['_n'] - 1).replace(0, np.nan)
    out.loc[sorted_frame.index, 'subject_pos_frac'] = pos_frac.to_numpy()
    return out


def build_target_matrix(
    target,
    frame,
    base_pred,
    raw_pred,
    graph_pred,
    graph_conf,
    context,
):
    x = pd.DataFrame(index=frame.index)
    x['base'] = base_pred[target].astype(float)
    x['raw'] = raw_pred[target].astype(float)
    x['graph'] = graph_pred[target].astype(float)
    x['graph_conf'] = graph_conf[target].astype(float)
    x['logit_base'] = logit(x['base'])
    x['logit_raw'] = logit(x['raw'])
    x['logit_graph'] = logit(x['graph'])
    x['raw_minus_base'] = x['raw'] - x['base']
    x['graph_minus_base'] = x['graph'] - x['base']
    x['abs_raw_minus_base'] = (x['raw'] - x['base']).abs()
    x['abs_graph_minus_base'] = (x['graph'] - x['base']).abs()
    x['raw_graph_agree'] = 1.0 - (x['raw'] - x['graph']).abs()
    x = pd.concat([x, context.reset_index(drop=True), calendar_features(frame).reset_index(drop=True)], axis=1)
    return x


def fit_meta_target(x_train, y, x_test, target):
    y = np.asarray(y, dtype=int)
    n_folds = int(min(5, np.bincount(y, minlength=2).min()))
    if n_folds < 2:
        p = float(np.clip(y.mean(), 0.02, 0.98))
        return np.full(len(x_train), p), np.full(len(x_test), p), {'constant': p}

    lr_oof = np.zeros(len(x_train), dtype=float)
    ext_oof = np.zeros(len(x_train), dtype=float)
    lr_test = np.zeros(len(x_test), dtype=float)
    ext_test = np.zeros(len(x_test), dtype=float)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        seed_lr_oof = np.zeros(len(x_train), dtype=float)
        seed_ext_oof = np.zeros(len(x_train), dtype=float)
        seed_lr_test = np.zeros(len(x_test), dtype=float)
        seed_ext_test = np.zeros(len(x_test), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
            x_tr = x_train.iloc[tr_idx]
            x_va = x_train.iloc[va_idx]
            y_tr = y[tr_idx]

            lr = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    C=0.32,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=2000,
                    class_weight='balanced',
                )),
            ])
            lr.fit(x_tr, y_tr)
            seed_lr_oof[va_idx] = lr.predict_proba(x_va)[:, 1]
            seed_lr_test += lr.predict_proba(x_test)[:, 1] / n_folds

            imp = SimpleImputer(strategy='median')
            x_tr_imp = imp.fit_transform(x_tr)
            x_va_imp = imp.transform(x_va)
            x_test_imp = imp.transform(x_test)
            ext = ExtraTreesClassifier(
                n_estimators=520,
                max_depth=4,
                min_samples_leaf=12,
                max_features=0.70,
                class_weight='balanced',
                random_state=seed + fold * 17,
                n_jobs=-1,
            )
            ext.fit(x_tr_imp, y_tr)
            seed_ext_oof[va_idx] = ext.predict_proba(x_va_imp)[:, 1]
            seed_ext_test += ext.predict_proba(x_test_imp)[:, 1] / n_folds

        lr_oof += seed_lr_oof / len(SEEDS)
        ext_oof += seed_ext_oof / len(SEEDS)
        lr_test += seed_lr_test / len(SEEDS)
        ext_test += seed_ext_test / len(SEEDS)

    meta_oof = clip_prob(0.70 * lr_oof + 0.30 * ext_oof)
    meta_test = clip_prob(0.70 * lr_test + 0.30 * ext_test)
    diagnostics = {
        'lr_loss': target_logloss(y, lr_oof),
        'extra_loss': target_logloss(y, ext_oof),
        'meta_loss': target_logloss(y, meta_oof),
        'feature_count': int(x_train.shape[1]),
    }
    return meta_oof, meta_test, diagnostics


def blend_frames(base, other, weights):
    out = base.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        if weight <= 0:
            continue
        out[target] = clip_prob((1.0 - weight) * base[target] + weight * other[target])
    return out


def adaptive_weights(train, base_oof, meta_oof):
    weights = {}
    diagnostics = {}
    for target in TARGETS:
        base_loss = target_logloss(train[target], base_oof[target])
        meta_loss = target_logloss(train[target], meta_oof[target])
        gain = base_loss - meta_loss
        if gain > 0.035:
            weight = 0.45
        elif gain > 0.020:
            weight = 0.32
        elif gain > 0.010:
            weight = 0.22
        elif gain > 0.004:
            weight = 0.12
        elif gain > 0.000:
            weight = 0.05
        else:
            weight = 0.0
        weights[target] = weight
        diagnostics[target] = {
            'base_loss': base_loss,
            'meta_loss': meta_loss,
            'gain_base_minus_meta': gain,
            'weight': weight,
        }
    return weights, diagnostics


def describe_vs_base(pred, base):
    pred_arr = pred[TARGETS].to_numpy(dtype=float)
    base_arr = base[TARGETS].to_numpy(dtype=float)
    diff = pred_arr - base_arr
    return {
        'corr_vs_base': float(np.corrcoef(pred_arr.ravel(), base_arr.ravel())[0, 1]),
        'mad_vs_base': float(np.mean(np.abs(diff))),
        'max_abs_vs_base': float(np.max(np.abs(diff))),
        'means': {target: float(pred[target].mean()) for target in TARGETS},
    }


def save_candidate(name, train, sub, base_sub, oof_pred, sub_pred, policy):
    oof = make_keys(train)
    submission = make_keys(sub)
    for target in TARGETS:
        oof[target] = clip_prob(oof_pred[target])
        submission[target] = clip_prob(sub_pred[target])
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    oof_loss, per_target = evaluate_frame(train, oof)
    return {
        'name': name,
        'policy': policy,
        'oof_loss': oof_loss,
        'oof_per_target': per_target,
        'distribution_vs_base': describe_vs_base(submission, base_sub),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
    }


def main():
    ensure_dirs()
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_SAMPLE_PATH)
    base_tag, base_oof, base_sub, base_oof_path, base_sub_path = choose_base_prediction()
    raw_oof = load_frame(RAW_OOF)
    raw_sub = load_frame(RAW_SUB)
    graph_oof = load_frame(GRAPH_OOF)
    graph_sub = load_frame(GRAPH_SUB)
    graph_conf_oof = load_frame(GRAPH_CONF_OOF)
    graph_conf_sub = load_frame(GRAPH_CONF_SUB)

    print(f'[v50] base={base_tag}')
    train_context = build_train_context_features(train)
    test_context = build_test_context_features(train, sub)

    meta_oof = make_keys(train)
    meta_sub = make_keys(sub)
    target_diagnostics = {}
    feature_columns = {}
    for target in TARGETS:
        x_train = build_target_matrix(
            target,
            train,
            base_oof,
            raw_oof,
            graph_oof,
            graph_conf_oof,
            train_context[target],
        )
        x_test = build_target_matrix(
            target,
            sub,
            base_sub,
            raw_sub,
            graph_sub,
            graph_conf_sub,
            test_context[target],
        )
        oof_pred, sub_pred, diagnostics = fit_meta_target(x_train, train[target], x_test, target)
        meta_oof[target] = oof_pred
        meta_sub[target] = sub_pred
        target_diagnostics[target] = diagnostics
        feature_columns[target] = x_train.columns.tolist()
        print(f'[v50] {target}: meta_oof={diagnostics["meta_loss"]:.6f} lr={diagnostics["lr_loss"]:.6f} ext={diagnostics["extra_loss"]:.6f}')

    meta_name = 'v50_sequence_meta_raw'
    meta_oof_path = OOF_DIR / f'oof_{meta_name}.csv'
    meta_sub_path = SUB_DIR / f'submission_{meta_name}.csv'
    meta_oof.to_csv(meta_oof_path, index=False)
    meta_sub.to_csv(meta_sub_path, index=False)

    base_loss, base_per_target = evaluate_frame(train, base_oof)
    raw_loss, raw_per_target = evaluate_frame(train, raw_oof)
    graph_loss, graph_per_target = evaluate_frame(train, graph_oof)
    meta_loss, meta_per_target = evaluate_frame(train, meta_oof)

    candidates = [{
        'name': meta_name,
        'policy': {'type': 'meta_raw'},
        'oof_loss': meta_loss,
        'oof_per_target': meta_per_target,
        'distribution_vs_base': describe_vs_base(meta_sub, base_sub),
        'oof_path': str(meta_oof_path),
        'submission': str(meta_sub_path),
    }]

    adapt_w, adapt_diag = adaptive_weights(train, base_oof, meta_oof)
    blend_specs = {
        'v50_base_meta_safe': {
            'Q1': 0.10, 'Q2': 0.06, 'Q3': 0.05,
            'S1': 0.12, 'S2': 0.12, 'S3': 0.05, 'S4': 0.06,
        },
        'v50_base_meta_mid': {
            'Q1': 0.20, 'Q2': 0.10, 'Q3': 0.06,
            'S1': 0.22, 'S2': 0.20, 'S3': 0.06, 'S4': 0.08,
        },
        'v50_base_meta_q1s1s2strong': {
            'Q1': 0.34, 'Q2': 0.12, 'Q3': 0.04,
            'S1': 0.38, 'S2': 0.34, 'S3': 0.04, 'S4': 0.06,
        },
        'v50_base_meta_weak_targets': {
            'Q1': 0.06, 'Q2': 0.08, 'Q3': 0.18,
            'S1': 0.06, 'S2': 0.06, 'S3': 0.16, 'S4': 0.16,
        },
        'v50_base_meta_oofadaptive': adapt_w,
    }
    for name, weights in blend_specs.items():
        oof_pred = blend_frames(base_oof, meta_oof, weights)
        sub_pred = blend_frames(base_sub, meta_sub, weights)
        candidates.append(save_candidate(name, train, sub, base_sub, oof_pred, sub_pred, {
            'type': 'base_meta_target_blend',
            'base_tag': base_tag,
            'weights': weights,
        }))

    candidates = sorted(candidates, key=lambda item: item['oof_loss'])
    summary = {
        'exp_tag': 'v50_sequence_meta_blender',
        'base': {
            'tag': base_tag,
            'oof_path': base_oof_path,
            'submission_path': base_sub_path,
            'oof_loss': base_loss,
            'oof_per_target': base_per_target,
        },
        'raw': {'oof_loss': raw_loss, 'oof_per_target': raw_per_target},
        'graph': {'oof_loss': graph_loss, 'oof_per_target': graph_per_target},
        'meta_raw': {
            'oof_loss': meta_loss,
            'oof_per_target': meta_per_target,
            'oof_path': str(meta_oof_path),
            'submission_path': str(meta_sub_path),
            'target_diagnostics': target_diagnostics,
        },
        'adaptive_weight_diagnostics': adapt_diag,
        'feature_columns': feature_columns,
        'pseudo_block_lengths': PSEUDO_BLOCK_LENGTHS,
        'candidates': candidates,
        'notes': [
            'Meta OOF uses stratified folds over pseudo-hidden context features; it is still a proxy.',
            'If all meta blends are worse than base, continue v48 strength search and redesign graph features.',
            'If weak-target or adaptive blend helps public, graph/context is useful as routing signal rather than direct prediction.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v50_sequence_meta_blender.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('[v50] base_oof:', f'{base_loss:.6f}')
    print('[v50] raw_oof:', f'{raw_loss:.6f}')
    print('[v50] graph_oof:', f'{graph_loss:.6f}')
    print('[v50] meta_oof:', f'{meta_loss:.6f}')
    print('[v50] top candidates by OOF:')
    for item in candidates[:8]:
        print(' ', item['name'], f"oof={item['oof_loss']:.6f}", item['submission'])
    print('[v50] summary:', summary_path)


if __name__ == '__main__':
    main()

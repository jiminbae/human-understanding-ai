# v53: split-aware transductive sequence posterior layer.
#
# v48's best public result came from a policy layer over strong existing
# predictions, not from a bigger single classifier.  This script keeps that
# spirit but makes the policy aware of the actual train/test date blocks:
#   1. train a small posterior router on pseudo-hidden subject blocks,
#   2. run a target-wise HMM smoother over each full subject sequence using
#      train labels as hard observations and test predictions as emissions,
#   3. write conservative role-aware blends on top of the public-confirmed v48
#      base.
#
# It does not retrain raw sensor models; it consumes existing OOF/submission
# files from v47/v48/v50/v51 and produces v53 candidates.
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore', category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']
SOURCE_NAMES = ['base', 'raw', 'meta', 'guard']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
LOG_DIR = OUTPUTS_DIR / 'log'

RAW_TAG = 'v47_hourgrid_subject_state_residual_raw'
META_TAG = 'v50_sequence_meta_raw'
GUARD_TAGS = [
    'v51_meta_oofadaptive_capped',
    'v51_meta_raw_agreement_gate_loose',
    'v51_meta_guarded_public_safe_direct',
]

DEFAULT_BASE_TAGS = [
    'v48_target_delta_scaled_avg430_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg425_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg420_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg415_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg410_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg390_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg350_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg310_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg270_q2cap115_q3s3guard',
]

PSEUDO_BLOCK_LENGTHS = [1, 2, 3, 5, 8, 11, 14, 16]
SEEDS = [42, 2025, 777]
N_EXTRA_TREES = int(os.environ.get('V53_EXTRA_TREES', '360'))
MAX_PSEUDO_WINDOWS_PER_SUBJECT = int(os.environ.get('V53_MAX_WINDOWS_PER_SUBJECT', '0'))
EPS = 1e-7


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


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ['sleep_date', 'lifelog_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.reset_index(drop=True)


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


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


def choose_prediction(tag_env, default_tags, label):
    env_tag = os.environ.get(tag_env)
    tags = [env_tag] if env_tag else []
    tags.extend(default_tags)
    for tag in tags:
        if not tag:
            continue
        oof_path = OOF_DIR / f'oof_{tag}.csv'
        sub_path = SUB_DIR / f'submission_{tag}.csv'
        if oof_path.exists() and sub_path.exists():
            print(f'[v53] {label}={tag}')
            return tag, load_frame(oof_path), load_frame(sub_path), str(oof_path), str(sub_path)
    raise FileNotFoundError(f'No usable {label} prediction found.')


def optional_prediction(tag, fallback_oof, fallback_sub, label):
    oof_path = OOF_DIR / f'oof_{tag}.csv'
    sub_path = SUB_DIR / f'submission_{tag}.csv'
    if oof_path.exists() and sub_path.exists():
        print(f'[v53] {label}={tag}')
        return tag, load_frame(oof_path), load_frame(sub_path), str(oof_path), str(sub_path)
    print(f'[v53] {label} missing; using base as fallback')
    return 'base_fallback', fallback_oof.copy(), fallback_sub.copy(), None, None


def choose_guard_prediction(fallback_oof, fallback_sub):
    env_tag = os.environ.get('V53_GUARD_TAG')
    tags = [env_tag] if env_tag else []
    tags.extend(GUARD_TAGS)
    for tag in tags:
        if not tag:
            continue
        oof_path = OOF_DIR / f'oof_{tag}.csv'
        sub_path = SUB_DIR / f'submission_{tag}.csv'
        if oof_path.exists() and sub_path.exists():
            print(f'[v53] guard={tag}')
            return tag, load_frame(oof_path), load_frame(sub_path), str(oof_path), str(sub_path)
    print('[v53] guard missing; using base as fallback')
    return 'base_fallback', fallback_oof.copy(), fallback_sub.copy(), None, None


def build_profiles(train, sub):
    profiles = {}
    for subject_id in sorted(train['subject_id'].unique()):
        combined = pd.concat([
            train.loc[train['subject_id'] == subject_id, ['sleep_date']].assign(kind='T'),
            sub.loc[sub['subject_id'] == subject_id, ['sleep_date']].assign(kind='X'),
        ]).sort_values('sleep_date')
        runs = []
        for row in combined.itertuples(index=False):
            if not runs or runs[-1]['kind'] != row.kind:
                runs.append({'kind': row.kind, 'n': 1})
            else:
                runs[-1]['n'] += 1
        x_runs = [run['n'] for run in runs if run['kind'] == 'X']
        profiles[subject_id] = {
            'runs': runs,
            'x_runs': x_runs,
            'n_x_runs': len(x_runs),
            'is_fragmented': len(x_runs) >= 5,
            'interior_x_total': int(sum(x_runs[:-1])) if x_runs else 0,
            'tail_x_run': int(x_runs[-1]) if x_runs else 0,
        }
    return profiles


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
    return (
        seq
        .sort_values(['subject_id', 'sleep_date', 'lifelog_date', '_split'])
        .reset_index(drop=True)
    )


def neighbor_positions(known_positions, pos):
    prev_positions = known_positions[known_positions < pos]
    next_positions = known_positions[known_positions > pos]
    prev_pos = int(prev_positions[-1]) if len(prev_positions) else None
    next_pos = int(next_positions[0]) if len(next_positions) else None
    return prev_pos, next_pos


def weighted_visible_mean(seq, known_positions, target, pos, bandwidth):
    if len(known_positions) == 0:
        return np.nan
    row_date = seq.loc[pos, 'sleep_date']
    labels = seq.loc[known_positions, target].to_numpy(dtype=float)
    gaps = np.array([
        abs((row_date - seq.loc[k, 'sleep_date']).days)
        for k in known_positions
    ], dtype=float)
    weights = np.exp(-gaps / float(bandwidth))
    denom = max(float(weights.sum()), EPS)
    return float(np.dot(weights, labels) / denom)


def recent_mean(seq, known_positions, target, pos, side, max_count=3):
    if side == 'prev':
        positions = known_positions[known_positions < pos][-max_count:]
    else:
        positions = known_positions[known_positions > pos][:max_count]
    if len(positions) == 0:
        return np.nan
    return float(seq.loc[positions, target].mean())


def context_record(seq, known_positions, pos, target, global_mean, subject_mean):
    row = seq.loc[pos]
    subject_id = row['subject_id']
    prior = smoothed_subject_mean(subject_id, target, global_mean, subject_mean)
    prev_pos, next_pos = neighbor_positions(known_positions, pos)

    prev_label = np.nan
    next_label = np.nan
    dist_prev = np.nan
    dist_next = np.nan
    pos_frac = np.nan
    invdist = np.nan

    if prev_pos is not None:
        prev_label = float(seq.loc[prev_pos, target])
        dist_prev = max(1.0, float((row['sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days))
    if next_pos is not None:
        next_label = float(seq.loc[next_pos, target])
        dist_next = max(1.0, float((seq.loc[next_pos, 'sleep_date'] - row['sleep_date']).days))
    if prev_pos is not None and next_pos is not None:
        span = max(1.0, float((seq.loc[next_pos, 'sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days))
        pos_frac = float((row['sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days) / span
        pos_frac = float(np.clip(pos_frac, 0.0, 1.0))
        w_prev = 1.0 / max(dist_prev, 1.0)
        w_next = 1.0 / max(dist_next, 1.0)
        invdist = float((w_prev * prev_label + w_next * next_label) / (w_prev + w_next))
    elif prev_pos is not None:
        invdist = prev_label
    elif next_pos is not None:
        invdist = next_label

    values = {
        'ctx_prior': prior,
        'ctx_visible_mean': float(seq.loc[known_positions, target].mean()) if len(known_positions) else prior,
        'ctx_has_prev': float(prev_pos is not None),
        'ctx_has_next': float(next_pos is not None),
        'ctx_is_interior': float(prev_pos is not None and next_pos is not None),
        'ctx_prev_label': prev_label,
        'ctx_next_label': next_label,
        'ctx_prev_next_mean': np.nanmean([prev_label, next_label]),
        'ctx_prev_next_absdiff': (
            abs(prev_label - next_label)
            if np.isfinite(prev_label) and np.isfinite(next_label)
            else np.nan
        ),
        'ctx_prev_next_agree': float(
            np.isfinite(prev_label)
            and np.isfinite(next_label)
            and abs(prev_label - next_label) < 1e-12
        ),
        'ctx_dist_prev': dist_prev,
        'ctx_dist_next': dist_next,
        'ctx_dist_min': np.nanmin([dist_prev, dist_next]),
        'ctx_pos_frac': pos_frac,
        'ctx_invdist_bidir': invdist,
        'ctx_exp_kernel_7': weighted_visible_mean(seq, known_positions, target, pos, 7),
        'ctx_exp_kernel_14': weighted_visible_mean(seq, known_positions, target, pos, 14),
        'ctx_exp_kernel_30': weighted_visible_mean(seq, known_positions, target, pos, 30),
        'ctx_prev_recent3': recent_mean(seq, known_positions, target, pos, 'prev'),
        'ctx_next_recent3': recent_mean(seq, known_positions, target, pos, 'next'),
    }
    for key, value in list(values.items()):
        if isinstance(value, float) and not np.isfinite(value):
            values[key] = np.nan
    return values


def calendar_record(frame, row_idx, subject_n_rows):
    row = frame.loc[row_idx]
    sleep_dow = float(row['sleep_date'].dayofweek)
    lifelog_dow = float(row['lifelog_date'].dayofweek)
    order = float(row.get('_subject_order', np.nan))
    denom = max(1.0, float(subject_n_rows - 1))
    return {
        'subject_num': float(str(row['subject_id']).replace('id', '')),
        'sleep_dow': sleep_dow,
        'sleep_is_weekend': float(sleep_dow >= 5),
        'sleep_dow_sin': float(np.sin(2 * np.pi * sleep_dow / 7)),
        'sleep_dow_cos': float(np.cos(2 * np.pi * sleep_dow / 7)),
        'lifelog_dow': lifelog_dow,
        'lifelog_dow_sin': float(np.sin(2 * np.pi * lifelog_dow / 7)),
        'lifelog_dow_cos': float(np.cos(2 * np.pi * lifelog_dow / 7)),
        'subject_pos_frac': order / denom,
    }


def prediction_record(pred_sources, row_idx, target):
    rec = {}
    for source_name, frame in pred_sources.items():
        value = float(frame.loc[row_idx, target])
        rec[f'{source_name}'] = value
        rec[f'logit_{source_name}'] = float(logit(value))
        for other in TARGETS:
            rec[f'{source_name}_{other}'] = float(frame.loc[row_idx, other])

    rec['raw_minus_base'] = rec['raw'] - rec['base']
    rec['meta_minus_base'] = rec['meta'] - rec['base']
    rec['guard_minus_base'] = rec['guard'] - rec['base']
    rec['abs_raw_minus_base'] = abs(rec['raw_minus_base'])
    rec['abs_meta_minus_base'] = abs(rec['meta_minus_base'])
    rec['abs_guard_minus_base'] = abs(rec['guard_minus_base'])
    rec['raw_meta_agree'] = float((rec['raw_minus_base'] * rec['meta_minus_base']) > 0)
    rec['raw_guard_agree'] = float((rec['raw_minus_base'] * rec['guard_minus_base']) > 0)
    rec['meta_guard_agree'] = float((rec['meta_minus_base'] * rec['guard_minus_base']) > 0)
    rec['sleep_cluster_base_mean'] = float(np.mean([rec['base_S1'], rec['base_S2'], rec['base_S3'], rec['base_S4']]))
    rec['survey_cluster_base_mean'] = float(np.mean([rec['base_Q1'], rec['base_Q2'], rec['base_Q3']]))
    rec['s2_s4_base_gap'] = rec['base_S2'] - rec['base_S4']
    rec['q2_q3_base_gap'] = rec['base_Q2'] - rec['base_Q3']
    rec['q1_s1_base_gap'] = rec['base_Q1'] - rec['base_S1']
    return rec


def role_record(length, pos_in_run, has_prev, has_next, is_fragmented):
    denom = max(1, length - 1)
    pos_frac = 0.5 if length == 1 else pos_in_run / denom
    return {
        'hidden_len': float(length),
        'hidden_pos_frac': float(pos_frac),
        'hidden_centered_pos': float(pos_frac - 0.5),
        'hidden_is_singleton': float(length == 1),
        'hidden_is_short': float(length <= 3),
        'hidden_is_long': float(length >= 8),
        'hidden_has_prev': float(has_prev),
        'hidden_has_next': float(has_next),
        'hidden_is_interior': float(has_prev and has_next),
        'hidden_is_tail': float(has_prev and not has_next),
        'hidden_is_prefix': float((not has_prev) and has_next),
        'subject_is_fragmented': float(is_fragmented),
    }


def add_subject_order(frame):
    out = frame.sort_values(['subject_id', 'sleep_date', 'lifelog_date']).copy()
    out['_subject_order'] = out.groupby('subject_id').cumcount()
    return out.sort_index()


def pseudo_windows_for_subject(n):
    windows = []
    lengths = [length for length in PSEUDO_BLOCK_LENGTHS if length < n]
    for length in lengths:
        for start in range(0, n - length + 1):
            windows.append((start, length))
    if MAX_PSEUDO_WINDOWS_PER_SUBJECT and len(windows) > MAX_PSEUDO_WINDOWS_PER_SUBJECT:
        rng = np.random.default_rng(42 + n)
        selected = sorted(rng.choice(len(windows), size=MAX_PSEUDO_WINDOWS_PER_SUBJECT, replace=False))
        windows = [windows[i] for i in selected]
    return windows


def build_pseudo_samples(train, pred_sources, target, profiles, global_mean, subject_mean):
    train_seq = prepare_sequence_frame(train)
    train_seq = add_subject_order(train_seq)
    rows = []
    for subject_id, grp in train_seq.groupby('subject_id', sort=False):
        grp = grp.sort_values('sleep_date').reset_index(drop=True)
        n = len(grp)
        subject_is_fragmented = profiles[subject_id]['is_fragmented']
        for start, length in pseudo_windows_for_subject(n):
            hidden_positions = np.arange(start, start + length, dtype=int)
            hidden_set = set(hidden_positions.tolist())
            known_positions = np.array([pos for pos in range(n) if pos not in hidden_set], dtype=int)
            if len(known_positions) == 0:
                continue
            has_prev = start > 0
            has_next = start + length < n
            for offset, pos in enumerate(hidden_positions):
                orig_idx = int(grp.loc[pos, '_orig_index'])
                rec = {
                    '_orig_index': orig_idx,
                    '_label': int(grp.loc[pos, target]),
                }
                rec.update(prediction_record(pred_sources, orig_idx, target))
                rec.update(context_record(grp, known_positions, int(pos), target, global_mean, subject_mean))
                rec.update(role_record(length, offset, has_prev, has_next, subject_is_fragmented))
                rec.update(calendar_record(grp, int(pos), n))
                rows.append(rec)
    return pd.DataFrame(rows)


def build_actual_test_features(train, sub, pred_sources, target, profiles, global_mean, subject_mean):
    seq = prepare_sequence_frame(train, sub)
    seq = add_subject_order(seq)
    rows = [None] * len(sub)
    role_rows = [None] * len(sub)

    for subject_id, grp in seq.groupby('subject_id', sort=False):
        grp = grp.sort_values('sleep_date').reset_index(drop=True)
        known_positions = grp.index[grp['_split'] == 'train'].to_numpy(dtype=int)
        is_fragmented = profiles[subject_id]['is_fragmented']

        run_start = None
        for pos, split in enumerate(grp['_split'].tolist() + ['sentinel']):
            if split == 'test' and run_start is None:
                run_start = pos
            if split != 'test' and run_start is not None:
                run_end = pos
                run_positions = list(range(run_start, run_end))
                length = len(run_positions)
                has_prev = len(known_positions[known_positions < run_start]) > 0
                has_next = len(known_positions[known_positions > run_end - 1]) > 0
                for offset, row_pos in enumerate(run_positions):
                    sub_idx = int(grp.loc[row_pos, '_orig_index'])
                    rec = {'_orig_index': sub_idx}
                    rec.update(prediction_record(pred_sources, sub_idx, target))
                    rec.update(context_record(grp, known_positions, row_pos, target, global_mean, subject_mean))
                    role = role_record(length, offset, has_prev, has_next, is_fragmented)
                    rec.update(role)
                    rec.update(calendar_record(grp, row_pos, len(grp)))
                    rows[sub_idx] = rec
                    role_rows[sub_idx] = {
                        'subject_id': subject_id,
                        'run_len': length,
                        'run_pos_frac': role['hidden_pos_frac'],
                        'is_interior': bool(role['hidden_is_interior']),
                        'is_tail': bool(role['hidden_is_tail']),
                        'is_fragmented_subject': bool(is_fragmented),
                    }
                run_start = None

    feature_df = pd.DataFrame(rows)
    role_df = pd.DataFrame(role_rows)
    return feature_df, role_df


def fit_models(x_train, y_train):
    lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            C=0.22,
            penalty='l2',
            solver='lbfgs',
            max_iter=2500,
            class_weight='balanced',
        )),
    ])
    ext = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', ExtraTreesClassifier(
            n_estimators=N_EXTRA_TREES,
            max_depth=5,
            min_samples_leaf=16,
            max_features=0.70,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )),
    ])
    lr.fit(x_train, y_train)
    ext.fit(x_train, y_train)
    return lr, ext


def predict_models(models, x):
    lr, ext = models
    lr_pred = lr.predict_proba(x)[:, 1]
    ext_pred = ext.predict_proba(x)[:, 1]
    return clip_prob(0.68 * lr_pred + 0.32 * ext_pred), lr_pred, ext_pred


def fit_posterior_target(train, samples, test_features, target):
    drop_cols = ['_orig_index', '_label']
    feature_cols = [col for col in samples.columns if col not in drop_cols]
    x_samples = samples[feature_cols]
    y_samples = samples['_label'].to_numpy(dtype=int)
    orig = samples['_orig_index'].to_numpy(dtype=int)
    x_test = test_features[feature_cols]

    row_y = train[target].to_numpy(dtype=int)
    n_folds = int(min(5, np.bincount(row_y, minlength=2).min()))
    oof_sum = np.zeros(len(train), dtype=float)
    oof_count = np.zeros(len(train), dtype=float)
    fold_losses = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (row_tr_idx, row_va_idx) in enumerate(skf.split(np.arange(len(train)), row_y), start=1):
            va_mask = np.isin(orig, row_va_idx)
            tr_mask = ~va_mask
            models = fit_models(x_samples.loc[tr_mask], y_samples[tr_mask])
            pred, _, _ = predict_models(models, x_samples.loc[va_mask])
            va_orig = orig[va_mask]
            for row_idx, value in zip(va_orig, pred):
                oof_sum[row_idx] += value
                oof_count[row_idx] += 1.0

            row_pred = pd.Series(pred).groupby(va_orig).mean()
            fold_row_idx = row_pred.index.to_numpy(dtype=int)
            fold_losses.append(target_logloss(row_y[fold_row_idx], row_pred.to_numpy(dtype=float)))

    posterior_oof = np.divide(
        oof_sum,
        np.maximum(oof_count, 1.0),
        out=np.full(len(train), float(row_y.mean()), dtype=float),
        where=oof_count > 0,
    )
    final_models = fit_models(x_samples, y_samples)
    posterior_test, lr_test, ext_test = predict_models(final_models, x_test)
    diagnostics = {
        'target': target,
        'pseudo_samples': int(len(samples)),
        'feature_count': int(len(feature_cols)),
        'oof_loss': target_logloss(row_y, posterior_oof),
        'fold_loss_mean': float(np.mean(fold_losses)) if fold_losses else None,
        'fold_loss_std': float(np.std(fold_losses)) if fold_losses else None,
        'test_lr_ext_corr': float(np.corrcoef(lr_test, ext_test)[0, 1]),
    }
    return clip_prob(posterior_oof), clip_prob(posterior_test), diagnostics


def build_posterior(train, sub, train_sources, test_sources, profiles):
    global_mean, subject_mean = subject_target_priors(train)
    posterior_oof = make_keys(train)
    posterior_sub = make_keys(sub)
    role_df = None
    diagnostics = {}

    for target in TARGETS:
        print(f'\n[v53] posterior target={target}')
        samples = build_pseudo_samples(
            train,
            train_sources,
            target,
            profiles,
            global_mean,
            subject_mean,
        )
        test_features, target_role_df = build_actual_test_features(
            train,
            sub,
            test_sources,
            target,
            profiles,
            global_mean,
            subject_mean,
        )
        if role_df is None:
            role_df = target_role_df
        oof_pred, sub_pred, diag = fit_posterior_target(train, samples, test_features, target)
        posterior_oof[target] = oof_pred
        posterior_sub[target] = sub_pred
        diagnostics[target] = diag
        print(
            f'[v53] {target}: samples={diag["pseudo_samples"]} '
            f'features={diag["feature_count"]} oof={diag["oof_loss"]:.6f}'
        )

    return posterior_oof, posterior_sub, role_df, diagnostics


def estimate_transition(train, target, smoothing=1.25):
    counts = np.full((2, 2), float(smoothing), dtype=float)
    for _, grp in train.sort_values(['subject_id', 'sleep_date']).groupby('subject_id', sort=False):
        y = grp[target].to_numpy(dtype=int)
        dates = grp['sleep_date'].tolist()
        for i in range(len(grp) - 1):
            gap = max(1, int((dates[i + 1] - dates[i]).days))
            if gap <= 2:
                counts[y[i], y[i + 1]] += 1.0
    trans = counts / counts.sum(axis=1, keepdims=True)
    # Keep the Markov prior useful but not so sharp that it overwrites emissions.
    trans = 0.88 * trans + 0.12 * np.array([[0.5, 0.5], [0.5, 0.5]])
    return trans


def transition_power(trans, gap):
    steps = max(1, min(7, int(gap)))
    out = np.eye(2)
    for _ in range(steps):
        out = out @ trans
    return out


def forward_backward(emissions, transitions):
    n = len(emissions)
    alpha = np.zeros((n, 2), dtype=float)
    beta = np.ones((n, 2), dtype=float)

    alpha[0] = emissions[0] / max(float(emissions[0].sum()), EPS)
    for i in range(1, n):
        alpha[i] = (alpha[i - 1] @ transitions[i - 1]) * emissions[i]
        alpha[i] /= max(float(alpha[i].sum()), EPS)

    for i in range(n - 2, -1, -1):
        beta[i] = transitions[i] @ (emissions[i + 1] * beta[i + 1])
        beta[i] /= max(float(beta[i].sum()), EPS)

    posterior = alpha * beta
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), EPS)
    return posterior[:, 1]


def sequence_posterior_for_target(train, sub, emission_sub, target):
    trans = estimate_transition(train, target)
    seq = prepare_sequence_frame(train, sub)
    out = np.zeros(len(sub), dtype=float)

    for _, grp in seq.groupby('subject_id', sort=False):
        grp = grp.sort_values('sleep_date').reset_index(drop=True)
        emissions = []
        dates = grp['sleep_date'].tolist()
        transitions = []
        for _, row in grp.iterrows():
            if row['_split'] == 'train':
                y = int(row[target])
                p = 0.985 if y == 1 else 0.015
            else:
                p = float(emission_sub.loc[int(row['_orig_index']), target])
            emissions.append([1.0 - p, p])
        emissions = np.asarray(emissions, dtype=float)
        for i in range(len(grp) - 1):
            gap = max(1, int((dates[i + 1] - dates[i]).days))
            transitions.append(transition_power(trans, gap))
        post = forward_backward(emissions, transitions) if len(grp) > 1 else emissions[:, 1]
        for pos, (_, row) in enumerate(grp.iterrows()):
            if row['_split'] == 'test':
                out[int(row['_orig_index'])] = post[pos]
    return clip_prob(out)


def build_sequence_crf(train, sub, posterior_sub):
    crf_sub = make_keys(sub)
    trans_diag = {}
    for target in TARGETS:
        crf_sub[target] = sequence_posterior_for_target(train, sub, posterior_sub, target)
        trans_diag[target] = estimate_transition(train, target).round(6).tolist()
    return crf_sub, trans_diag


def blend_constant(base, other, weights):
    out = base.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        if weight <= 0:
            continue
        out[target] = clip_prob((1.0 - weight) * base[target] + weight * other[target])
    return out


def role_scales(role_df, mode):
    if mode == 'safe':
        interior, fragmented, tail = 1.15, 0.95, 0.65
    elif mode == 'sensor':
        interior, fragmented, tail = 1.05, 1.00, 0.95
    elif mode == 'latent':
        interior, fragmented, tail = 1.10, 1.00, 0.85
    else:
        interior, fragmented, tail = 1.0, 1.0, 1.0

    scale = np.ones(len(role_df), dtype=float)
    is_tail = role_df['is_tail'].to_numpy(dtype=bool)
    is_interior = role_df['is_interior'].to_numpy(dtype=bool)
    is_fragmented = role_df['is_fragmented_subject'].to_numpy(dtype=bool)
    scale[is_tail] = tail
    scale[is_interior] = interior
    scale[is_interior & is_fragmented] = fragmented
    return scale


def blend_role_aware(base, other, weights, role_df, mode):
    out = base.copy()
    scale = role_scales(role_df, mode)
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        if weight <= 0:
            continue
        row_weight = np.clip(weight * scale, 0.0, 0.45)
        out[target] = clip_prob(
            (1.0 - row_weight) * base[target].to_numpy(dtype=float)
            + row_weight * other[target].to_numpy(dtype=float)
        )
    return out


def apply_s2_s4_coupling(train, frame, s4_weight=0.18, s2_weight=0.04):
    out = frame.copy()
    offset_s4_from_s2 = float(logit(train['S4'].mean()) - logit(train['S2'].mean()))
    offset_s2_from_s4 = -offset_s4_from_s2
    s4_from_s2 = logit(out['S2']) + offset_s4_from_s2
    s2_from_s4 = logit(out['S4']) + offset_s2_from_s4
    out['S4'] = clip_prob(sigmoid((1.0 - s4_weight) * logit(out['S4']) + s4_weight * s4_from_s2))
    out['S2'] = clip_prob(sigmoid((1.0 - s2_weight) * logit(out['S2']) + s2_weight * s2_from_s4))
    return out


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
        'oof_proxy_loss': oof_loss,
        'oof_proxy_per_target': per_target,
        'distribution_vs_base': describe_vs_base(submission, base_sub),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
    }


def write_raw_frame(name, train, sub, oof_pred, sub_pred):
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof_pred.to_csv(oof_path, index=False)
    sub_pred.to_csv(sub_path, index=False)
    return str(oof_path), str(sub_path)


def main():
    ensure_dirs()
    log_path = LOG_DIR / 'run_v53_split_aware_sequence_posterior.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print('[v53] starting split-aware sequence posterior')
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_SAMPLE_PATH)
    train = add_subject_order(train)
    sub = add_subject_order(sub)

    base_tag, base_oof, base_sub, base_oof_path, base_sub_path = choose_prediction(
        'V53_BASE_TAG',
        DEFAULT_BASE_TAGS,
        'base',
    )
    raw_tag, raw_oof, raw_sub, raw_oof_path, raw_sub_path = optional_prediction(
        RAW_TAG,
        base_oof,
        base_sub,
        'raw',
    )
    meta_tag, meta_oof, meta_sub, meta_oof_path, meta_sub_path = optional_prediction(
        META_TAG,
        base_oof,
        base_sub,
        'meta',
    )
    guard_tag, guard_oof, guard_sub, guard_oof_path, guard_sub_path = choose_guard_prediction(
        base_oof,
        base_sub,
    )

    train_sources = {
        'base': base_oof,
        'raw': raw_oof,
        'meta': meta_oof,
        'guard': guard_oof,
    }
    test_sources = {
        'base': base_sub,
        'raw': raw_sub,
        'meta': meta_sub,
        'guard': guard_sub,
    }

    profiles = build_profiles(train, sub)
    posterior_oof_file = OOF_DIR / 'oof_v53_split_aware_posterior_raw.csv'
    posterior_sub_file = SUB_DIR / 'submission_v53_split_aware_posterior_raw.csv'
    reuse_posterior = (
        os.environ.get('V53_REUSE_POSTERIOR', '1') == '1'
        and posterior_oof_file.exists()
        and posterior_sub_file.exists()
    )
    if reuse_posterior:
        print('[v53] reusing existing posterior raw files')
        posterior_oof = load_frame(posterior_oof_file)
        posterior_sub = load_frame(posterior_sub_file)
        global_mean, subject_mean = subject_target_priors(train)
        _, role_df = build_actual_test_features(
            train,
            sub,
            test_sources,
            'Q1',
            profiles,
            global_mean,
            subject_mean,
        )
        posterior_diag = {'reused_from': str(posterior_sub_file)}
        posterior_oof_path = str(posterior_oof_file)
        posterior_sub_path = str(posterior_sub_file)
    else:
        posterior_oof, posterior_sub, role_df, posterior_diag = build_posterior(
            train,
            sub,
            train_sources,
            test_sources,
            profiles,
        )
        posterior_oof_path, posterior_sub_path = write_raw_frame(
            'v53_split_aware_posterior_raw',
            train,
            sub,
            posterior_oof,
            posterior_sub,
        )
    posterior_loss, posterior_per_target = evaluate_frame(train, posterior_oof)

    crf_sub, transition_diag = build_sequence_crf(train, sub, posterior_sub)
    _, crf_sub_path = write_raw_frame(
        'v53_sequence_crf_raw',
        train,
        sub,
        posterior_oof,
        crf_sub,
    )
    crf_coupled_sub = apply_s2_s4_coupling(train, crf_sub, s4_weight=0.20, s2_weight=0.05)
    posterior_coupled_oof = apply_s2_s4_coupling(train, posterior_oof, s4_weight=0.18, s2_weight=0.04)

    candidates = []
    candidate_specs = [
        (
            'v53_crf_safe',
            {'Q1': 0.10, 'Q2': 0.05, 'Q3': 0.03, 'S1': 0.12, 'S2': 0.10, 'S3': 0.04, 'S4': 0.05},
            crf_sub,
            posterior_oof,
            'safe',
        ),
        (
            'v53_q1s1s2_sensor_posterior',
            {'Q1': 0.18, 'Q2': 0.02, 'Q3': 0.00, 'S1': 0.18, 'S2': 0.16, 'S3': 0.00, 'S4': 0.03},
            crf_sub,
            posterior_oof,
            'sensor',
        ),
        (
            'v53_s2s4_latent_bridge',
            {'Q1': 0.08, 'Q2': 0.03, 'Q3': 0.00, 'S1': 0.08, 'S2': 0.12, 'S3': 0.02, 'S4': 0.16},
            crf_coupled_sub,
            posterior_coupled_oof,
            'latent',
        ),
        (
            'v53_public_mid_probe',
            {'Q1': 0.14, 'Q2': 0.04, 'Q3': 0.02, 'S1': 0.14, 'S2': 0.12, 'S3': 0.03, 'S4': 0.08},
            crf_coupled_sub,
            posterior_coupled_oof,
            'safe',
        ),
    ]

    for name, weights, sub_source, oof_source, mode in candidate_specs:
        oof_pred = blend_constant(base_oof, oof_source, weights)
        sub_pred = blend_role_aware(base_sub, sub_source, weights, role_df, mode)
        candidates.append(save_candidate(
            name,
            train,
            sub,
            base_sub,
            oof_pred,
            sub_pred,
            {
                'type': 'role_aware_base_to_sequence_posterior_blend',
                'base_tag': base_tag,
                'weights': weights,
                'role_scale_mode': mode,
                'source': 'crf_coupled' if 'latent' in name or 'mid' in name else 'crf',
            },
        ))

    candidates = sorted(candidates, key=lambda item: item['oof_proxy_loss'])
    base_loss, base_per_target = evaluate_frame(train, base_oof)
    raw_loss, raw_per_target = evaluate_frame(train, raw_oof)
    meta_loss, meta_per_target = evaluate_frame(train, meta_oof)
    guard_loss, guard_per_target = evaluate_frame(train, guard_oof)

    role_summary = {
        'n_rows': int(len(role_df)),
        'tail_rows': int(role_df['is_tail'].sum()),
        'interior_rows': int(role_df['is_interior'].sum()),
        'fragmented_subject_rows': int(role_df['is_fragmented_subject'].sum()),
        'run_len_counts': {
            str(k): int(v)
            for k, v in role_df['run_len'].value_counts().sort_index().items()
        },
    }
    summary = {
        'exp_tag': 'v53_split_aware_sequence_posterior',
        'inputs': {
            'base': {'tag': base_tag, 'oof': base_oof_path, 'submission': base_sub_path},
            'raw': {'tag': raw_tag, 'oof': raw_oof_path, 'submission': raw_sub_path},
            'meta': {'tag': meta_tag, 'oof': meta_oof_path, 'submission': meta_sub_path},
            'guard': {'tag': guard_tag, 'oof': guard_oof_path, 'submission': guard_sub_path},
        },
        'source_oof': {
            'base': {'loss': base_loss, 'per_target': base_per_target},
            'raw': {'loss': raw_loss, 'per_target': raw_per_target},
            'meta': {'loss': meta_loss, 'per_target': meta_per_target},
            'guard': {'loss': guard_loss, 'per_target': guard_per_target},
            'posterior_raw': {'loss': posterior_loss, 'per_target': posterior_per_target},
        },
        'posterior': {
            'oof_path': posterior_oof_path,
            'submission_path': posterior_sub_path,
            'target_diagnostics': posterior_diag,
        },
        'crf': {
            'submission_path': crf_sub_path,
            'transition_matrices': transition_diag,
        },
        'profiles': profiles,
        'role_summary': role_summary,
        'candidates': candidates,
        'notes': [
            'Candidate OOF is a proxy: train rows use posterior OOF without the final transductive CRF pass.',
            'Submission predictions use actual test block roles and train labels as hard HMM observations.',
            'The safest daily submission order is usually crf_safe, q1s1s2_sensor_posterior, s2s4_latent_bridge.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v53_split_aware_sequence_posterior.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('\n[v53] source OOF')
    print(f'  base={base_loss:.6f} raw={raw_loss:.6f} meta={meta_loss:.6f} guard={guard_loss:.6f}')
    print(f'  posterior_raw={posterior_loss:.6f}')
    print('[v53] role summary:', role_summary)
    print('[v53] candidates by proxy OOF:')
    for item in candidates:
        print(
            f'  {item["name"]}: proxy={item["oof_proxy_loss"]:.6f} '
            f'mad={item["distribution_vs_base"]["mad_vs_base"]:.6f} '
            f'sub={item["submission"]}'
        )
    print(f'[v53] summary={summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

# v31: explicit subject-level interpolation stack on top of v23/v29.
#   - No GBM retraining.
#   - Uses v23/v29 OOF + submission predictions as stable base models.
#   - Adds simple prev/next/nearest/distance-weighted target interpolation.
#   - Writes raw stack plus conservative target-gated candidates.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

EXP_TAG = 'v31_interpolation_stack'
RUN_LOG_PATH = LOG_DIR / f'run_{EXP_TAG}.log'

V23_OOF = OOF_DIR / 'oof_v23_stability_filter.csv'
V29_OOF = OOF_DIR / 'oof_v29_bidirectional_target_history.csv'
V23_SUB = SUB_DIR / 'submission_v23_stability_filter.csv'
V29_SUB = SUB_DIR / 'submission_v29_bidirectional_target_history.csv'


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


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=float), 1e-5, 1 - 1e-5)


def logit(x):
    x = clip_prob(x)
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_inputs():
    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_PATH)
    for df in [train, sub]:
        df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
        df['sleep_date'] = pd.to_datetime(df['sleep_date'])

    oof23 = pd.read_csv(V23_OOF)
    oof29 = pd.read_csv(V29_OOF)
    sub23 = pd.read_csv(V23_SUB)
    sub29 = pd.read_csv(V29_SUB)
    for df in [oof23, oof29, sub23, sub29]:
        df['lifelog_date'] = pd.to_datetime(df['lifelog_date'])
        df['sleep_date'] = pd.to_datetime(df['sleep_date'])
    return train, sub, oof23, oof29, sub23, sub29


def build_subject_histories(train, target):
    histories = {}
    cols = ['subject_id', 'lifelog_date', target]
    for sid, grp in train[cols].sort_values(['subject_id', 'lifelog_date']).groupby('subject_id'):
        histories[sid] = {
            'dates': grp['lifelog_date'].to_numpy(),
            'labels': grp[target].astype(float).to_numpy(),
        }
    return histories


def interpolation_features(train, query, target, exclude_self):
    histories = build_subject_histories(train, target)
    global_mean = float(train[target].mean())
    rows = []
    for sid, d in query[['subject_id', 'lifelog_date']].itertuples(index=False):
        d64 = np.datetime64(pd.Timestamp(d))
        row = {
            'interp_subject_mean': np.nan,
            'interp_prev1': np.nan,
            'interp_next1': np.nan,
            'interp_nearest': np.nan,
            'interp_invw': np.nan,
            'interp_local14': np.nan,
            'interp_prev_dist': np.nan,
            'interp_next_dist': np.nan,
            'interp_has_both': 0.0,
            'interp_gap': np.nan,
            'interp_conf': 0.0,
        }
        if sid not in histories:
            row['interp_subject_mean'] = global_mean
            row['interp_invw'] = global_mean
            row['interp_nearest'] = global_mean
            row['interp_local14'] = global_mean
            rows.append(row)
            continue

        dates = histories[sid]['dates']
        labels = histories[sid]['labels']
        mask = np.ones(len(labels), dtype=bool)
        if exclude_self:
            mask &= dates != d64
        h_dates = dates[mask]
        h_labels = labels[mask]
        if len(h_labels) == 0:
            row['interp_subject_mean'] = global_mean
            row['interp_invw'] = global_mean
            row['interp_nearest'] = global_mean
            row['interp_local14'] = global_mean
            rows.append(row)
            continue

        row['interp_subject_mean'] = float(np.mean(h_labels))
        left = np.searchsorted(h_dates, d64, side='left')
        right = np.searchsorted(h_dates, d64, side='right')
        past_labels = h_labels[:left]
        future_labels = h_labels[right:]

        if len(past_labels) > 0:
            prev1 = float(past_labels[-1])
            prev_dist = float((d64 - h_dates[left - 1]) / np.timedelta64(1, 'D'))
            row['interp_prev1'] = prev1
            row['interp_prev_dist'] = prev_dist
        if len(future_labels) > 0:
            next1 = float(future_labels[0])
            next_dist = float((h_dates[right] - d64) / np.timedelta64(1, 'D'))
            row['interp_next1'] = next1
            row['interp_next_dist'] = next_dist

        local_mask = np.abs((h_dates - d64) / np.timedelta64(1, 'D')) <= 14
        local_labels = h_labels[local_mask]
        if len(local_labels) > 0:
            row['interp_local14'] = float(np.mean(local_labels))
        else:
            row['interp_local14'] = row['interp_subject_mean']

        has_prev = not np.isnan(row['interp_prev1'])
        has_next = not np.isnan(row['interp_next1'])
        if has_prev and has_next:
            row['interp_has_both'] = 1.0
            row['interp_gap'] = row['interp_next1'] - row['interp_prev1']
            prev_w = 1.0 / (row['interp_prev_dist'] + 1.0)
            next_w = 1.0 / (row['interp_next_dist'] + 1.0)
            row['interp_invw'] = (row['interp_prev1'] * prev_w + row['interp_next1'] * next_w) / (prev_w + next_w)
            row['interp_nearest'] = row['interp_prev1'] if row['interp_prev_dist'] <= row['interp_next_dist'] else row['interp_next1']
            row['interp_conf'] = abs(row['interp_invw'] - 0.5) * 2.0
        elif has_prev:
            row['interp_invw'] = row['interp_prev1']
            row['interp_nearest'] = row['interp_prev1']
            row['interp_conf'] = abs(row['interp_prev1'] - 0.5) * 2.0
        elif has_next:
            row['interp_invw'] = row['interp_next1']
            row['interp_nearest'] = row['interp_next1']
            row['interp_conf'] = abs(row['interp_next1'] - 0.5) * 2.0
        else:
            row['interp_invw'] = row['interp_subject_mean']
            row['interp_nearest'] = row['interp_subject_mean']
            row['interp_conf'] = abs(row['interp_subject_mean'] - 0.5) * 2.0

        rows.append(row)
    return pd.DataFrame(rows, index=query.index)


def build_meta_matrix(base23, base29, interp_df, target):
    X = pd.DataFrame(index=interp_df.index)
    X['v23_logit'] = logit(base23[target])
    X['v29_logit'] = logit(base29[target])
    X['v29_minus_v23'] = X['v29_logit'] - X['v23_logit']
    for col in interp_df.columns:
        X[col] = interp_df[col].values
    prob_cols = [
        'interp_subject_mean', 'interp_prev1', 'interp_next1',
        'interp_nearest', 'interp_invw', 'interp_local14',
    ]
    for col in prob_cols:
        X[f'{col}_logit'] = logit(X[col].fillna(X['interp_subject_mean'].fillna(0.5)))
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_stack(train, sub, oof23, oof29, sub23, sub29):
    oof_stack = train[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    sub_stack = sub[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    interp_oof_map = {}
    interp_test_map = {}
    per_target = {}

    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        print(f'\n=== {target} ===')
        interp_oof = interpolation_features(train, train, target, exclude_self=True)
        interp_test = interpolation_features(train, sub, target, exclude_self=False)
        interp_oof_map[target] = interp_oof
        interp_test_map[target] = interp_test

        X = build_meta_matrix(oof23, oof29, interp_oof, target)
        X_test = build_meta_matrix(sub23, sub29, interp_test, target)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_oof = np.zeros(len(train))
        meta_test = np.zeros(len(sub))
        for tr_idx, val_idx in skf.split(X, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.15, penalty='l2', solver='lbfgs', max_iter=1000),
            )
            model.fit(X.iloc[tr_idx], y[tr_idx])
            meta_oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
            meta_test += model.predict_proba(X_test)[:, 1] / skf.n_splits

        interp_prob_oof = clip_prob(interp_oof['interp_invw'])
        print(f"  v23={log_loss(y, clip_prob(oof23[target])):.5f}"
              f"  v29={log_loss(y, clip_prob(oof29[target])):.5f}"
              f"  interp={log_loss(y, interp_prob_oof):.5f}"
              f"  stack={log_loss(y, clip_prob(meta_oof)):.5f}")
        oof_stack[target] = clip_prob(meta_oof)
        sub_stack[target] = clip_prob(meta_test)
        per_target[target] = float(log_loss(y, clip_prob(meta_oof)))

    return oof_stack, sub_stack, interp_oof_map, interp_test_map, per_target


def save_target_gated_candidates(train, sub, oof23, oof29, sub23, sub29, oof_stack, sub_stack):
    candidates = {}

    # Conservative public-informed candidate: preserve proven v29 q234, only test stack where OOF is clearly better.
    candidates['v31_stack_q234_s2hold'] = {
        'Q1': ('v29', 1.0),
        'Q2': ('v29', 1.0),
        'Q3': ('v29', 1.0),
        'S1': ('v23', 1.0),
        'S2': ('v23', 1.0),
        'S3': ('v23', 1.0),
        'S4': ('v29', 1.0),
    }

    # Stack probe: Q targets and S4 use v31 stack, S1/S3 held, S2 held.
    candidates['v31_stack_q234_probe'] = {
        'Q1': ('stack', 1.0),
        'Q2': ('stack', 1.0),
        'Q3': ('stack', 1.0),
        'S1': ('v23', 1.0),
        'S2': ('v23', 1.0),
        'S3': ('v23', 1.0),
        'S4': ('stack', 1.0),
    }

    # Half-stack probe: less movement than raw stack.
    candidates['v31_stack_q234_half'] = {
        'Q1': ('blend_v29_stack', 0.5),
        'Q2': ('blend_v29_stack', 0.5),
        'Q3': ('blend_v29_stack', 0.5),
        'S1': ('v23', 1.0),
        'S2': ('v23', 1.0),
        'S3': ('v23', 1.0),
        'S4': ('blend_v29_stack', 0.5),
    }

    saved = []
    for name, spec in candidates.items():
        out = sub[['subject_id', 'sleep_date', 'lifelog_date']].copy()
        oof_out = train[['subject_id', 'sleep_date', 'lifelog_date']].copy()
        for target in TARGETS:
            mode, weight = spec[target]
            if mode == 'v23':
                sub_pred = sub23[target]
                oof_pred = oof23[target]
            elif mode == 'v29':
                sub_pred = sub29[target]
                oof_pred = oof29[target]
            elif mode == 'stack':
                sub_pred = sub_stack[target]
                oof_pred = oof_stack[target]
            elif mode == 'blend_v29_stack':
                sub_pred = weight * sub29[target] + (1.0 - weight) * sub_stack[target]
                oof_pred = weight * oof29[target] + (1.0 - weight) * oof_stack[target]
            else:
                raise ValueError(mode)
            out[target] = clip_prob(sub_pred)
            oof_out[target] = clip_prob(oof_pred)

        sub_path = SUB_DIR / f'submission_{name}.csv'
        oof_path = OOF_DIR / f'oof_{name}.csv'
        out.to_csv(sub_path, index=False)
        oof_out.to_csv(oof_path, index=False)
        losses = {t: log_loss(train[t], oof_out[t]) for t in TARGETS}
        print(f'\n{name}: OOF={np.mean(list(losses.values())):.6f}  {losses}')
        print(f'  saved: {sub_path}')
        saved.append(str(sub_path))
    return saved


def main():
    ensure_dirs()
    log_f = open(RUN_LOG_PATH, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG}...')
    train, sub, oof23, oof29, sub23, sub29 = load_inputs()
    oof_stack, sub_stack, _, _, per_target = train_stack(train, sub, oof23, oof29, sub23, sub29)

    oof_path = OOF_DIR / f'oof_{EXP_TAG}.csv'
    sub_path = SUB_DIR / f'submission_{EXP_TAG}.csv'
    oof_stack.to_csv(oof_path, index=False)
    sub_stack.to_csv(sub_path, index=False)
    print(f'\nRaw stack OOF={np.mean(list(per_target.values())):.6f}')
    print(f'oof saved: {oof_path}')
    print(f'submission saved: {sub_path}')

    saved = save_target_gated_candidates(train, sub, oof23, oof29, sub23, sub29, oof_stack, sub_stack)
    summary = {
        'exp_tag': EXP_TAG,
        'raw_oof': float(np.mean(list(per_target.values()))),
        'per_target': per_target,
        'saved_submissions': saved,
    }
    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'summary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

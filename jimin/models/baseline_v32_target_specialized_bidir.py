# v32: target-specialized v23/v29 policy from public feedback.
#   - No new GBM training.
#   - v23 remains the anchor for S1/S3.
#   - v29 simple bidirectional target-history model is used for Q1/Q2/Q3/S4.
#   - S2 is blended with a tunable v29 weight, defaulting to the current best 0.65.
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

EXP_TAG = os.environ.get('V32_EXP_TAG', 'v32_target_specialized_bidir')
S2_V29_WEIGHT = float(os.environ.get('V32_S2_V29_WEIGHT', '0.65'))

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
    return np.clip(np.asarray(x, dtype=float), 0.02, 0.98)


def logit(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def blend_prob(a, b, b_weight):
    return clip_prob((1.0 - b_weight) * np.asarray(a, dtype=float) + b_weight * np.asarray(b, dtype=float))


def blend_logit(a, b, b_weight):
    return clip_prob(sigmoid((1.0 - b_weight) * logit(a) + b_weight * logit(b)))


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['lifelog_date', 'sleep_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def build_policy_frame(keys, pred23, pred29, s2_weight, blend_fn):
    out = keys.copy()
    for target in TARGETS:
        if target in ['Q1', 'Q2', 'Q3', 'S4']:
            out[target] = clip_prob(pred29[target])
        elif target in ['S1', 'S3']:
            out[target] = clip_prob(pred23[target])
        elif target == 'S2':
            out[target] = blend_fn(pred23[target], pred29[target], s2_weight)
        else:
            raise ValueError(target)
    return out


def evaluate(train, oof):
    per_target = {target: log_loss(train[target], np.clip(oof[target], 1e-6, 1 - 1e-6))
                  for target in TARGETS}
    return float(np.mean(list(per_target.values()))), per_target


def describe_vs_v23(name, sub, sub23):
    ref = sub23[TARGETS].to_numpy().ravel()
    arr = sub[TARGETS].to_numpy().ravel()
    corr = float(np.corrcoef(ref, arr)[0, 1])
    mad = float(np.mean(np.abs(ref - arr)))
    max_abs = float(np.max(np.abs(ref - arr)))
    means = {target: float(sub[target].mean()) for target in TARGETS}
    print(f'{name}: corr_vs_v23={corr:.6f} mad={mad:.6f} max={max_abs:.6f}')
    print(f'  means={means}')
    return {'corr_vs_v23': corr, 'mad_vs_v23': mad, 'max_abs_vs_v23': max_abs, 'means': means}


def save_candidate(name, train, sub_keys, oof23, oof29, sub23, sub29, s2_weight, blend_fn):
    oof = build_policy_frame(train[['subject_id', 'sleep_date', 'lifelog_date']], oof23, oof29, s2_weight, blend_fn)
    submission = build_policy_frame(sub_keys, sub23, sub29, s2_weight, blend_fn)
    oof_total, per_target = evaluate(train, oof)

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    print(f'\n{name}: OOF={oof_total:.6f}')
    print(f'  per_target={per_target}')
    print(f'  saved={sub_path}')
    dist = describe_vs_v23(name, submission, sub23)
    return {
        'name': name,
        's2_v29_weight': s2_weight,
        'oof': oof_total,
        'per_target': per_target,
        'submission': str(sub_path),
        'oof_path': str(oof_path),
        'distribution': dist,
    }


def main():
    ensure_dirs()
    log_path = LOG_DIR / f'run_{EXP_TAG}.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    if not (0.0 <= S2_V29_WEIGHT <= 1.0):
        raise ValueError('V32_S2_V29_WEIGHT must be in [0, 1]')

    print(f'Starting {EXP_TAG}...')
    print(f'  S2_V29_WEIGHT={S2_V29_WEIGHT}')

    train = load_frame(TRAIN_PATH)
    sub_sample = load_frame(SUB_PATH)
    oof23 = load_frame(V23_OOF)
    oof29 = load_frame(V29_OOF)
    sub23 = load_frame(V23_SUB)
    sub29 = load_frame(V29_SUB)
    sub_keys = sub_sample[['subject_id', 'sleep_date', 'lifelog_date']].copy()

    summaries = []
    summaries.append(save_candidate(
        EXP_TAG, train, sub_keys, oof23, oof29, sub23, sub29, S2_V29_WEIGHT, blend_prob))

    # Same target policy, but S2 mixed in logit space. This is a small new axis to test after prob-space saturates.
    summaries.append(save_candidate(
        f'{EXP_TAG}_s2logit', train, sub_keys, oof23, oof29, sub23, sub29, S2_V29_WEIGHT, blend_logit))

    # Local sweep files around the current best; useful when public feedback says the peak moved.
    for weight in [0.625, 0.675, 0.70]:
        suffix = str(weight).replace('.', 'p')
        summaries.append(save_candidate(
            f'{EXP_TAG}_s2w{suffix}', train, sub_keys, oof23, oof29, sub23, sub29, weight, blend_prob))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        's2_v29_weight': S2_V29_WEIGHT,
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

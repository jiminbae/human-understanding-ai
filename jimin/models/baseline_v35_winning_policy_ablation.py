# v35: ablation around the v34 public-winning policy.
#   - Anchor: v32_target_specialized_bidir_s2w0p675.
#   - Winning policy from public feedback: Q1=Pass1, Q3/S4=Pass2, others=v32.
#   - This script isolates each target contribution and creates small blend probes.
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

EXP_TAG = os.environ.get('V35_EXP_TAG', 'v35_winning_policy_ablation')

V32_OOF = OOF_DIR / 'oof_v32_target_specialized_bidir_s2w0p675.csv'
V33_PASS1_OOF = OOF_DIR / 'oof_v33_long_history_cross_target_pass1.csv'
V33_PASS2_OOF = OOF_DIR / 'oof_v33_long_history_cross_target_pass2.csv'

V32_SUB = SUB_DIR / 'submission_v32_target_specialized_bidir_s2w0p675.csv'
V33_PASS1_SUB = SUB_DIR / 'submission_v33_long_history_cross_target_pass1.csv'
V33_PASS2_SUB = SUB_DIR / 'submission_v33_long_history_cross_target_pass2.csv'


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


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=float), 0.02, 0.98)


def blend_prob(anchor, candidate, candidate_weight):
    return clip_prob(
        (1.0 - candidate_weight) * np.asarray(anchor, dtype=float)
        + candidate_weight * np.asarray(candidate, dtype=float))


def evaluate(train, oof):
    per_target = {
        target: log_loss(train[target].values, np.clip(oof[target].values, 1e-7, 1 - 1e-7))
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def describe_vs_anchor(submission, anchor):
    ref = anchor[TARGETS].to_numpy().ravel()
    arr = submission[TARGETS].to_numpy().ravel()
    return {
        'corr_vs_v32': float(np.corrcoef(ref, arr)[0, 1]),
        'mad_vs_v32': float(np.mean(np.abs(ref - arr))),
        'max_abs_vs_v32': float(np.max(np.abs(ref - arr))),
        'means': {target: float(submission[target].mean()) for target in TARGETS},
    }


def build_candidate(keys, anchor, pass1, pass2, policy):
    out = keys.copy()
    for target in TARGETS:
        source, weight = policy.get(target, ('v32', 1.0))
        weight = float(weight)
        if source == 'v32':
            out[target] = clip_prob(anchor[target])
        elif source == 'pass1':
            out[target] = clip_prob(pass1[target])
        elif source == 'pass2':
            out[target] = clip_prob(pass2[target])
        elif source == 'blend_pass1':
            out[target] = blend_prob(anchor[target], pass1[target], weight)
        elif source == 'blend_pass2':
            out[target] = blend_prob(anchor[target], pass2[target], weight)
        else:
            raise ValueError(f'Unknown source for {target}: {source}')
    return out


def save_candidate(name, train, keys, anchor_oof, pass1_oof, pass2_oof, anchor_sub, pass1_sub, pass2_sub, policy):
    oof = build_candidate(train[['subject_id', 'sleep_date', 'lifelog_date']], anchor_oof, pass1_oof, pass2_oof, policy)
    submission = build_candidate(keys, anchor_sub, pass1_sub, pass2_sub, policy)
    oof_total, per_target = evaluate(train, oof)

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    dist = describe_vs_anchor(submission, anchor_sub)
    print(f'\n{name}: OOF={oof_total:.6f}')
    print(f'  per_target={per_target}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')
    return {
        'name': name,
        'policy': policy,
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

    print(f'Starting {EXP_TAG}...')
    train = load_frame(TRAIN_PATH)
    sub_sample = load_frame(SUB_PATH)
    keys = sub_sample[['subject_id', 'sleep_date', 'lifelog_date']].copy()

    anchor_oof = load_frame(V32_OOF)
    pass1_oof = load_frame(V33_PASS1_OOF)
    pass2_oof = load_frame(V33_PASS2_OOF)
    anchor_sub = load_frame(V32_SUB)
    pass1_sub = load_frame(V33_PASS1_SUB)
    pass2_sub = load_frame(V33_PASS2_SUB)

    policies = {
        # Single-target contribution probes.
        f'{EXP_TAG}_q1_pass1_only': {
            'Q1': ('pass1', 1.0),
        },
        f'{EXP_TAG}_q3_pass2_only': {
            'Q3': ('pass2', 1.0),
        },
        f'{EXP_TAG}_s4_pass2_only': {
            'S4': ('pass2', 1.0),
        },
        # Pair ablations around the v34 winner.
        f'{EXP_TAG}_q1p1_q3p2': {
            'Q1': ('pass1', 1.0),
            'Q3': ('pass2', 1.0),
        },
        f'{EXP_TAG}_q1p1_s4p2': {
            'Q1': ('pass1', 1.0),
            'S4': ('pass2', 1.0),
        },
        f'{EXP_TAG}_q3s4p2': {
            'Q3': ('pass2', 1.0),
            'S4': ('pass2', 1.0),
        },
        # Public-winning v34 policy, kept here as the new anchor candidate.
        f'{EXP_TAG}_q1p1_q3s4p2': {
            'Q1': ('pass1', 1.0),
            'Q3': ('pass2', 1.0),
            'S4': ('pass2', 1.0),
        },
        # Conservative/nearby probes.
        f'{EXP_TAG}_q1blend50_q3s4p2': {
            'Q1': ('blend_pass1', 0.50),
            'Q3': ('pass2', 1.0),
            'S4': ('pass2', 1.0),
        },
        f'{EXP_TAG}_q1p1_q3blend70_s4p2': {
            'Q1': ('pass1', 1.0),
            'Q3': ('blend_pass2', 0.70),
            'S4': ('pass2', 1.0),
        },
        f'{EXP_TAG}_q1p1_q3p2_s4blend70': {
            'Q1': ('pass1', 1.0),
            'Q3': ('pass2', 1.0),
            'S4': ('blend_pass2', 0.70),
        },
        f'{EXP_TAG}_q1p1_q3s4p2_s2p1_05': {
            'Q1': ('pass1', 1.0),
            'Q3': ('pass2', 1.0),
            'S2': ('blend_pass1', 0.05),
            'S4': ('pass2', 1.0),
        },
        f'{EXP_TAG}_q1p1_q3s4p2_s2p1_10': {
            'Q1': ('pass1', 1.0),
            'Q3': ('pass2', 1.0),
            'S2': ('blend_pass1', 0.10),
            'S4': ('pass2', 1.0),
        },
        f'{EXP_TAG}_q1p1_q3s4p2_s3p2_25': {
            'Q1': ('pass1', 1.0),
            'Q3': ('pass2', 1.0),
            'S3': ('blend_pass2', 0.25),
            'S4': ('pass2', 1.0),
        },
    }

    summaries = []
    for name, policy in policies.items():
        summaries.append(save_candidate(
            name, train, keys, anchor_oof, pass1_oof, pass2_oof,
            anchor_sub, pass1_sub, pass2_sub, policy))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'anchor': 'v32_target_specialized_bidir_s2w0p675',
        'known_public_best_policy': 'q1p1_q3s4p2',
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

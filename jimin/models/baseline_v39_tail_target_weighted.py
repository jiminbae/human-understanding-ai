# v39: target-weighted tail refinement after v38 public validation.
#   - v38 confirmed that tail-only treatment transfers to public LB.
#   - v39 keeps the v34 winner for interior rows and learns nothing new from
#     fine-grained row gates; it only varies tail blend strength by target.
#   - Conservative candidates stay near the public-validated v38 w25 anchor,
#     while stronger probes reflect target-specific tail proxy evidence.
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

EXP_TAG = os.environ.get('V39_EXP_TAG', 'v39_tail_target_weighted')

OOF_PATHS = {
    'v28a': OOF_DIR / 'oof_v28a_fwd_only.csv',
    'v28b': OOF_DIR / 'oof_v28b_pseudo85_fwd.csv',
    'v34': OOF_DIR / 'oof_v35_winning_policy_ablation_q1p1_q3s4p2.csv',
}
SUB_PATHS = {
    'v28a': SUB_DIR / 'submission_v28a_fwd_only.csv',
    'v28b': SUB_DIR / 'submission_v28b_pseudo85_fwd.csv',
    'v34': SUB_DIR / 'submission_v35_winning_policy_ablation_q1p1_q3s4p2.csv',
}


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
    return tail_mask


def build_proxy_masks(train, test, min_visible_each_side=6, min_visible_tail=8):
    middle = pd.Series(False, index=train.index)
    tail = pd.Series(False, index=train.index)
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

        idx = grp.index.to_numpy()
        n = len(idx)
        middle_len = min(x_runs[0], max(0, n - 2 * min_visible_each_side))
        if middle_len > 0:
            start = int(np.clip(
                (n - middle_len) // 2,
                min_visible_each_side,
                n - middle_len - min_visible_each_side,
            ))
            middle.loc[idx[start:start + middle_len]] = True

        tail_len = min(x_runs[-1], max(0, n - min_visible_tail))
        if tail_len > 0:
            tail.loc[idx[-tail_len:]] = True
    return middle, tail


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
        'corr_vs_v34': float(np.corrcoef(ref, arr)[0, 1]),
        'mad_vs_v34': float(np.mean(np.abs(ref - arr))),
        'max_abs_vs_v34': float(np.max(np.abs(ref - arr))),
        'means': {target: float(submission[target].mean()) for target in TARGETS},
    }


def build_candidate(keys, frames, tail_mask, policy):
    out = keys.copy()
    usage = {}
    for target in TARGETS:
        out[target] = clip_prob(frames['v34'][target])
        spec = policy.get(target)
        if spec is None:
            usage[target] = 'v34'
            continue
        source, weight = spec
        weight = float(weight)
        out.loc[tail_mask, target] = clip_prob(
            (1.0 - weight) * frames['v34'].loc[tail_mask, target]
            + weight * frames[source].loc[tail_mask, target]
        )
        usage[target] = f'{source}@{weight:.2f}'
    return out, usage


def save_candidate(name, train, keys, oof_frames, sub_frames, middle_mask, proxy_tail_mask, actual_tail_mask, policy):
    oof, usage = build_candidate(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        oof_frames,
        proxy_tail_mask,
        policy,
    )
    submission, _ = build_candidate(keys, sub_frames, actual_tail_mask, policy)
    hybrid_mask = middle_mask | proxy_tail_mask
    hybrid_total, hybrid_per_target = evaluate(train, oof, hybrid_mask)
    middle_total, middle_per_target = evaluate(train, oof, middle_mask)
    tail_total, tail_per_target = evaluate(train, oof, proxy_tail_mask)

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    dist = describe_vs_anchor(submission, sub_frames['v34'])
    print(f'\n{name}: hybrid={hybrid_total:.6f} middle={middle_total:.6f} tail={tail_total:.6f}')
    print(f'  hybrid_per_target={hybrid_per_target}')
    print(f'  tail_per_target={tail_per_target}')
    print(f'  usage={usage}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')
    return {
        'name': name,
        'policy': policy,
        'hybrid_proxy': hybrid_total,
        'middle_proxy': middle_total,
        'tail_proxy': tail_total,
        'hybrid_per_target': hybrid_per_target,
        'middle_per_target': middle_per_target,
        'tail_per_target': tail_per_target,
        'usage': usage,
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
    train = load_frame(TRAIN_PATH).reset_index(drop=True)
    sub = load_frame(SUB_PATH).reset_index(drop=True)
    keys = sub[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    oof_frames = {name: load_frame(path) for name, path in OOF_PATHS.items()}
    sub_frames = {name: load_frame(path) for name, path in SUB_PATHS.items()}

    middle_mask, proxy_tail_mask = build_proxy_masks(train, sub)
    actual_tail_mask = build_actual_tail_mask(train, sub)
    print(f'proxy roles: middle={int(middle_mask.sum())} tail={int(proxy_tail_mask.sum())}')
    print(f'actual tail rows: {int(actual_tail_mask.sum())}')

    candidates = {
        # Public-validated anchor from v38.
        f'{EXP_TAG}_uniform25_repro': {
            'Q2': ('v28b', 0.25),
            'S1': ('v28b', 0.25),
            'S2': ('v28b', 0.25),
            'S3': ('v28b', 0.25),
            'S4': ('v28a', 0.25),
        },
        # Gentle target weighting: move only the clearest targets beyond w25.
        f'{EXP_TAG}_gentle': {
            'Q2': ('v28b', 0.25),
            'S1': ('v28b', 0.50),
            'S2': ('v28b', 0.50),
            'S3': ('v28b', 0.25),
            'S4': ('v28a', 0.50),
        },
        # Balanced candidate for the next public submission after uniform w50.
        f'{EXP_TAG}_balanced': {
            'Q2': ('v28b', 0.50),
            'S1': ('v28b', 0.75),
            'S2': ('v28b', 0.75),
            'S3': ('v28b', 0.50),
            'S4': ('v28a', 0.75),
        },
        # Proxy-optimal target weighting without touching Q1/Q3.
        f'{EXP_TAG}_proxy_target_opt': {
            'Q2': ('v28b', 0.75),
            'S1': ('v28b', 1.00),
            'S2': ('v28b', 1.00),
            'S3': ('v28b', 0.75),
            'S4': ('v28a', 1.00),
        },
    }

    summaries = []
    for name, policy in candidates.items():
        summaries.append(save_candidate(
            name,
            train,
            keys,
            oof_frames,
            sub_frames,
            middle_mask,
            proxy_tail_mask,
            actual_tail_mask,
            policy,
        ))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'proxy_role_counts': {
            'middle': int(middle_mask.sum()),
            'tail': int(proxy_tail_mask.sum()),
        },
        'actual_tail_rows': int(actual_tail_mask.sum()),
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

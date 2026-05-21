# v42: conservative interior source ablation on top of the public-best w40 tail anchor.
#   - Tail stays fixed at the validated v38 w40 correction.
#   - Richer interior proxies show the most stable reusable v30 gains on
#     Q2/S1/S2/S4, with Q3 as a more aggressive optional extension.
#   - This script asks whether those interior gains transfer to public LB before
#     any new interior specialist is trained.
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from jimin.analysis import pseudo_public_interior_profile_eval as interior_eval


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

EXP_TAG = os.environ.get('V42_EXP_TAG', 'v42_interior_source_ablation')

ANCHOR_OOF_PATH = OOF_DIR / 'oof_v38_block_role_aware_tail_conservative_w40.csv'
ANCHOR_SUB_PATH = SUB_DIR / 'submission_v38_block_role_aware_tail_conservative_w40.csv'
V30_OOF_PATH = OOF_DIR / 'oof_v30_bidir_history_extended.csv'
V30_SUB_PATH = SUB_DIR / 'submission_v30_bidir_history_extended.csv'


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


def build_actual_interior_mask(train, sub):
    interior_mask = np.zeros(len(sub), dtype=bool)
    for sid, grp in sub.groupby('subject_id', sort=True):
        train_dates = train.loc[train['subject_id'] == sid, 'sleep_date']
        for idx, sleep_date in grp['sleep_date'].items():
            interior_mask[idx] = bool((train_dates > sleep_date).any())
    return pd.Series(interior_mask, index=sub.index)


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


def build_candidate(keys, anchor, v30, interior_mask, policy):
    out = keys.copy()
    for target in TARGETS:
        out[target] = clip_prob(anchor[target])
        weight = policy.get(target)
        if weight is None:
            continue
        out.loc[interior_mask, target] = clip_prob(
            (1.0 - weight) * anchor.loc[interior_mask, target].to_numpy()
            + weight * v30.loc[interior_mask, target].to_numpy()
        )
    return out


def save_candidate(
    name,
    train,
    keys,
    anchor_oof,
    anchor_sub,
    v30_oof,
    v30_sub,
    simple_mask,
    fragmented_mask,
    all_mask,
    actual_interior_mask,
    policy,
):
    oof = build_candidate(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        anchor_oof,
        v30_oof,
        all_mask,
        policy,
    )
    submission = build_candidate(
        keys,
        anchor_sub,
        v30_sub,
        actual_interior_mask,
        policy,
    )

    all_total, all_per_target = evaluate(train, oof, all_mask)
    simple_total, simple_per_target = evaluate(train, oof, simple_mask)
    fragmented_total, fragmented_per_target = evaluate(train, oof, fragmented_mask)

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    dist = describe_vs_anchor(submission, anchor_sub)
    print(
        f'\n{name}: all={all_total:.6f} '
        f'simple={simple_total:.6f} fragmented={fragmented_total:.6f}'
    )
    print(f'  all_per_target={all_per_target}')
    print(f'  simple_per_target={simple_per_target}')
    print(f'  fragmented_per_target={fragmented_per_target}')
    print(f'  policy={policy}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')
    return {
        'name': name,
        'policy': policy,
        'all_interior_proxy': all_total,
        'simple_interior_proxy': simple_total,
        'fragmented_interior_proxy': fragmented_total,
        'all_per_target': all_per_target,
        'simple_per_target': simple_per_target,
        'fragmented_per_target': fragmented_per_target,
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
    anchor_oof = load_frame(ANCHOR_OOF_PATH).reset_index(drop=True)
    anchor_sub = load_frame(ANCHOR_SUB_PATH).reset_index(drop=True)
    v30_oof = load_frame(V30_OOF_PATH).reset_index(drop=True)
    v30_sub = load_frame(V30_SUB_PATH).reset_index(drop=True)

    profiles = interior_eval.build_profiles(train, sub)
    simple_mask, fragmented_mask, all_mask = interior_eval.build_interior_masks(train, profiles)
    actual_interior_mask = build_actual_interior_mask(train, sub)
    print(
        f'proxy rows: simple={int(simple_mask.sum())} '
        f'fragmented={int(fragmented_mask.sum())} all={int(all_mask.sum())}'
    )
    print(f'actual interior rows={int(actual_interior_mask.sum())}')

    candidates = {
        f'{EXP_TAG}_winner_repro': {},
        f'{EXP_TAG}_core_q2s1s2s4_w25': {
            'Q2': 0.25, 'S1': 0.25, 'S2': 0.25, 'S4': 0.25,
        },
        f'{EXP_TAG}_core_q2s1s2s4_w50': {
            'Q2': 0.50, 'S1': 0.50, 'S2': 0.50, 'S4': 0.50,
        },
        f'{EXP_TAG}_core_q2s1s2s4_full': {
            'Q2': 1.00, 'S1': 1.00, 'S2': 1.00, 'S4': 1.00,
        },
        f'{EXP_TAG}_coreplusq3_w25': {
            'Q2': 0.25, 'Q3': 0.25, 'S1': 0.25, 'S2': 0.25, 'S4': 0.25,
        },
        f'{EXP_TAG}_coreplusq3_w50': {
            'Q2': 0.50, 'Q3': 0.50, 'S1': 0.50, 'S2': 0.50, 'S4': 0.50,
        },
        f'{EXP_TAG}_coreplusq3_full': {
            'Q2': 1.00, 'Q3': 1.00, 'S1': 1.00, 'S2': 1.00, 'S4': 1.00,
        },
    }

    summaries = []
    for name, policy in candidates.items():
        summaries.append(save_candidate(
            name,
            train,
            keys,
            anchor_oof,
            anchor_sub,
            v30_oof,
            v30_sub,
            simple_mask,
            fragmented_mask,
            all_mask,
            actual_interior_mask,
            policy,
        ))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'anchor': str(ANCHOR_SUB_PATH),
        'proxy_rows': {
            'simple': int(simple_mask.sum()),
            'fragmented': int(fragmented_mask.sum()),
            'all': int(all_mask.sum()),
        },
        'actual_interior_rows': int(actual_interior_mask.sum()),
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

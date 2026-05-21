# v41: low-degree block-geometry-aware tail correction.
#   - v38 showed that tail rows need a different source family than interior rows.
#   - Public feedback placed the useful uniform correction near w=0.40.
#   - v41 keeps the same trusted tail experts, but varies one shared row weight
#     from observable block geometry instead of using target-wise weights.
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TAIL_SOURCES = {
    'Q2': 'v28b',
    'S1': 'v28b',
    'S2': 'v28b',
    'S3': 'v28b',
    'S4': 'v28a',
}

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

EXP_TAG = os.environ.get('V41_EXP_TAG', 'v41_block_geometry_tail')

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
    return pd.Series(tail_mask, index=sub.index)


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

        idx = grp.sort_values('sleep_date').index.to_numpy()
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


def build_split_profiles(train, sub):
    profiles = {}
    for sid in sorted(train['subject_id'].unique()):
        combined = pd.concat([
            train.loc[train['subject_id'] == sid, ['sleep_date']].assign(kind='T'),
            sub.loc[sub['subject_id'] == sid, ['sleep_date']].assign(kind='X'),
        ]).sort_values('sleep_date')
        runs = []
        for row in combined.itertuples(index=False):
            if not runs or runs[-1][0] != row.kind:
                runs.append([row.kind, 1])
            else:
                runs[-1][1] += 1
        x_runs = [n for kind, n in runs if kind == 'X']
        profiles[sid] = {
            'runs': runs,
            'x_runs': x_runs,
            'n_x_runs': len(x_runs),
            'tail_len': int(x_runs[-1]),
            'simple_tail': len(x_runs) <= 2,
            'fragmented_tail': len(x_runs) >= 5,
        }
    return profiles


def build_tail_context(frame, tail_mask, profiles):
    context = pd.DataFrame(index=frame.index)
    context['tail_len'] = 0
    context['tail_pos'] = np.nan
    context['tail_pos_frac'] = np.nan
    context['n_x_runs'] = 0
    context['simple_tail'] = False
    context['fragmented_tail'] = False

    for sid, grp in frame.groupby('subject_id', sort=True):
        ordered = grp.sort_values('sleep_date')
        tail_idx = ordered.index[tail_mask.loc[ordered.index].to_numpy()]
        n_tail = len(tail_idx)
        profile = profiles[sid]
        for pos, idx in enumerate(tail_idx):
            context.loc[idx, 'tail_len'] = profile['tail_len']
            context.loc[idx, 'tail_pos'] = pos
            context.loc[idx, 'tail_pos_frac'] = float(pos / max(1, n_tail - 1))
            context.loc[idx, 'n_x_runs'] = profile['n_x_runs']
            context.loc[idx, 'simple_tail'] = profile['simple_tail']
            context.loc[idx, 'fragmented_tail'] = profile['fragmented_tail']
    return context


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


def weights_uniform40(context, tail_mask):
    return pd.Series(np.where(tail_mask, 0.40, 0.0), index=context.index)


def weights_len_3bin(context, tail_mask):
    values = np.where(
        context['tail_len'] <= 5,
        0.25,
        np.where(context['tail_len'] >= 10, 0.50, 0.40),
    )
    return pd.Series(np.where(tail_mask, values, 0.0), index=context.index)


def weights_simple_frag(context, tail_mask):
    values = np.where(context['simple_tail'], 0.50, 0.25)
    return pd.Series(np.where(tail_mask, values, 0.0), index=context.index)


def weights_twofactor(context, tail_mask):
    strong = context['simple_tail'] & (context['tail_len'] >= 10)
    weak = context['fragmented_tail'] | (context['tail_len'] <= 5)
    values = np.where(strong, 0.55, np.where(weak, 0.25, 0.40))
    return pd.Series(np.where(tail_mask, values, 0.0), index=context.index)


def weights_gentle_geom(context, tail_mask):
    strong = context['simple_tail'] & (context['tail_len'] >= 10)
    weak = context['fragmented_tail'] | (context['tail_len'] <= 5)
    values = 0.40 + np.where(strong, 0.05, 0.0) - np.where(weak, 0.05, 0.0)
    return pd.Series(np.where(tail_mask, values, 0.0), index=context.index)


def build_candidate(keys, frames, tail_mask, weights):
    out = keys.copy()
    for target in TARGETS:
        out[target] = clip_prob(frames['v34'][target])
        source = TAIL_SOURCES.get(target)
        if source is None:
            continue
        out.loc[tail_mask, target] = clip_prob(
            (1.0 - weights.loc[tail_mask].to_numpy()) * frames['v34'].loc[tail_mask, target].to_numpy()
            + weights.loc[tail_mask].to_numpy() * frames[source].loc[tail_mask, target].to_numpy()
        )
    return out


def summarize_weights(weights, tail_mask, context):
    tail_weights = weights.loc[tail_mask]
    return {
        'mean': float(tail_weights.mean()),
        'unique': sorted(float(x) for x in tail_weights.unique()),
    }


def save_candidate(
    name,
    train,
    keys,
    oof_frames,
    sub_frames,
    middle_mask,
    proxy_tail_mask,
    actual_tail_mask,
    proxy_context,
    actual_context,
    weight_fn,
):
    proxy_weights = weight_fn(proxy_context, proxy_tail_mask)
    actual_weights = weight_fn(actual_context, actual_tail_mask)
    oof = build_candidate(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        oof_frames,
        proxy_tail_mask,
        proxy_weights,
    )
    submission = build_candidate(keys, sub_frames, actual_tail_mask, actual_weights)

    hybrid_mask = middle_mask | proxy_tail_mask
    hybrid_total, hybrid_per_target = evaluate(train, oof, hybrid_mask)
    middle_total, middle_per_target = evaluate(train, oof, middle_mask)
    tail_total, tail_per_target = evaluate(train, oof, proxy_tail_mask)

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    dist = describe_vs_anchor(submission, sub_frames['v34'])
    proxy_weight_summary = summarize_weights(proxy_weights, proxy_tail_mask, proxy_context)
    actual_weight_summary = summarize_weights(actual_weights, actual_tail_mask, actual_context)
    print(f'\n{name}: hybrid={hybrid_total:.6f} middle={middle_total:.6f} tail={tail_total:.6f}')
    print(f'  hybrid_per_target={hybrid_per_target}')
    print(f'  tail_per_target={tail_per_target}')
    print(f'  proxy_weights={proxy_weight_summary}')
    print(f'  actual_weights={actual_weight_summary}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')
    return {
        'name': name,
        'hybrid_proxy': hybrid_total,
        'middle_proxy': middle_total,
        'tail_proxy': tail_total,
        'hybrid_per_target': hybrid_per_target,
        'middle_per_target': middle_per_target,
        'tail_per_target': tail_per_target,
        'proxy_weights': proxy_weight_summary,
        'actual_weights': actual_weight_summary,
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
    profiles = build_split_profiles(train, sub)
    proxy_context = build_tail_context(train, proxy_tail_mask, profiles)
    actual_context = build_tail_context(sub, actual_tail_mask, profiles)

    print(f'proxy roles: middle={int(middle_mask.sum())} tail={int(proxy_tail_mask.sum())}')
    print(f'actual tail rows: {int(actual_tail_mask.sum())}')
    print(f'profiles={profiles}')

    candidates = {
        f'{EXP_TAG}_uniform40_repro': weights_uniform40,
        f'{EXP_TAG}_len3bin': weights_len_3bin,
        f'{EXP_TAG}_simplefrag': weights_simple_frag,
        f'{EXP_TAG}_gentle_geom': weights_gentle_geom,
        f'{EXP_TAG}_twofactor': weights_twofactor,
    }

    summaries = []
    for name, weight_fn in candidates.items():
        summaries.append(save_candidate(
            name,
            train,
            keys,
            oof_frames,
            sub_frames,
            middle_mask,
            proxy_tail_mask,
            actual_tail_mask,
            proxy_context,
            actual_context,
            weight_fn,
        ))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'tail_sources': TAIL_SOURCES,
        'profiles': profiles,
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

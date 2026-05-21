# v45: uncertainty-aware temporal smoothing.
#   - Label-only reconstruction was weak as a standalone model, but the test
#     split exposes each subject's nearby train labels around many hidden rows.
#   - This script keeps the public-best v38/w40 anchor and only nudges uncertain
#     interior predictions toward nearby subject labels when temporal evidence is
#     close and coherent.
#   - Tail rows are left untouched in the primary candidates because v38/w40 has
#     already validated the coarse tail correction.
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
from jimin.models import baseline_v38_block_role_aware as v38


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
EXP_TAG = os.environ.get('V45_EXP_TAG', 'v45_uncertainty_temporal_smoothing')

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


def build_disjoint_proxy_role_masks(train, profiles):
    interior = pd.Series(False, index=train.index)
    simple = pd.Series(False, index=train.index)
    fragmented = pd.Series(False, index=train.index)
    tail = pd.Series(False, index=train.index)

    for sid, grp in train.groupby('subject_id', sort=True):
        profile = profiles[sid]
        idx = grp.sort_values('sleep_date').index.to_numpy()
        lengths = list(profile['x_runs'])
        if not lengths:
            continue

        hidden_total = int(sum(lengths))
        visible_total = len(idx) - hidden_total
        raw_gaps = profile['t_runs'][:len(lengths)]
        gaps = interior_eval.proportional_gaps(visible_total, raw_gaps)

        cursor = 0
        for block_i, block_len in enumerate(lengths):
            cursor += gaps[block_i]
            block = idx[cursor:cursor + block_len].tolist()
            cursor += block_len
            if block_i == len(lengths) - 1:
                tail.loc[block] = True
            else:
                interior.loc[block] = True
                if profile['is_simple']:
                    simple.loc[block] = True
                if profile['is_fragmented']:
                    fragmented.loc[block] = True
    return simple, fragmented, interior, tail


def nearest_indices(dates, query_date):
    d64 = np.datetime64(pd.Timestamp(query_date))
    left = np.searchsorted(dates, d64, side='left')
    prev_idx = left - 1 if left > 0 else None
    next_idx = left if left < len(dates) else None
    return prev_idx, next_idx, d64


def kernel_estimate(dates, labels, query_d64, subj_prior, bandwidth, shrink=1.0):
    if len(labels) == 0:
        return subj_prior, 0.0, np.inf
    gaps = np.abs((dates - query_d64) / np.timedelta64(1, 'D')).astype(float)
    weights = np.exp(-gaps / float(bandwidth))
    weight_sum = float(weights.sum())
    estimate = float((np.dot(weights, labels) + shrink * subj_prior) / (weight_sum + shrink))
    nearest_gap = float(gaps.min()) if len(gaps) else np.inf
    confidence = np.exp(-nearest_gap / 21.0) * min(1.0, weight_sum / 2.0)
    return estimate, float(confidence), nearest_gap


def build_temporal_features(history, query, target):
    global_prior = float(history[target].mean())
    if not np.isfinite(global_prior):
        global_prior = 0.5

    history_by_subject = {}
    for sid, grp in history.sort_values(['subject_id', 'sleep_date']).groupby('subject_id', sort=True):
        labels = grp[target].to_numpy(dtype=float)
        history_by_subject[sid] = {
            'dates': grp['sleep_date'].to_numpy(),
            'labels': labels,
            'prior': float(np.mean(labels)) if len(labels) else global_prior,
        }

    rows = []
    for idx, row in query[['subject_id', 'sleep_date']].iterrows():
        sid = row['subject_id']
        if sid not in history_by_subject:
            rows.append({
                'index': idx,
                'subject_prior': global_prior,
                'prev_label': np.nan,
                'next_label': np.nan,
                'prev_gap': np.inf,
                'next_gap': np.inf,
                'has_prev': 0.0,
                'has_next': 0.0,
                'agree_est': global_prior,
                'agree_conf': 0.0,
                'bracket_est': global_prior,
                'bracket_conf': 0.0,
                'kernel7_est': global_prior,
                'kernel7_conf': 0.0,
                'kernel14_est': global_prior,
                'kernel14_conf': 0.0,
                'kernel30_est': global_prior,
                'kernel30_conf': 0.0,
            })
            continue

        h = history_by_subject[sid]
        dates = h['dates']
        labels = h['labels']
        subj_prior = h['prior']
        prev_idx, next_idx, d64 = nearest_indices(dates, row['sleep_date'])

        prev_label = labels[prev_idx] if prev_idx is not None else np.nan
        next_label = labels[next_idx] if next_idx is not None else np.nan
        prev_gap = float((d64 - dates[prev_idx]) / np.timedelta64(1, 'D')) if prev_idx is not None else np.inf
        next_gap = float((dates[next_idx] - d64) / np.timedelta64(1, 'D')) if next_idx is not None else np.inf
        has_prev = float(prev_idx is not None)
        has_next = float(next_idx is not None)

        if prev_idx is not None and next_idx is not None:
            denom = max(1.0, prev_gap + next_gap)
            bracket_est = float((next_gap * prev_label + prev_gap * next_label) / denom)
            bracket_conf = float(np.exp(-max(prev_gap, next_gap) / 28.0))
            if prev_label == next_label:
                agree_est = float(0.90 * prev_label + 0.10 * subj_prior)
                agree_conf = float(np.exp(-max(prev_gap, next_gap) / 28.0))
            else:
                agree_est = bracket_est
                agree_conf = 0.0
        elif prev_idx is not None:
            bracket_est = float(0.80 * prev_label + 0.20 * subj_prior)
            bracket_conf = float(0.45 * np.exp(-prev_gap / 28.0))
            agree_est = bracket_est
            agree_conf = 0.0
        elif next_idx is not None:
            bracket_est = float(0.80 * next_label + 0.20 * subj_prior)
            bracket_conf = float(0.45 * np.exp(-next_gap / 28.0))
            agree_est = bracket_est
            agree_conf = 0.0
        else:
            bracket_est = subj_prior
            bracket_conf = 0.0
            agree_est = subj_prior
            agree_conf = 0.0

        kernel7_est, kernel7_conf, _ = kernel_estimate(dates, labels, d64, subj_prior, 7)
        kernel14_est, kernel14_conf, _ = kernel_estimate(dates, labels, d64, subj_prior, 14)
        kernel30_est, kernel30_conf, _ = kernel_estimate(dates, labels, d64, subj_prior, 30)

        rows.append({
            'index': idx,
            'subject_prior': subj_prior,
            'prev_label': prev_label,
            'next_label': next_label,
            'prev_gap': prev_gap,
            'next_gap': next_gap,
            'has_prev': has_prev,
            'has_next': has_next,
            'agree_est': agree_est,
            'agree_conf': agree_conf,
            'bracket_est': bracket_est,
            'bracket_conf': bracket_conf,
            'kernel7_est': kernel7_est,
            'kernel7_conf': kernel7_conf,
            'kernel14_est': kernel14_est,
            'kernel14_conf': kernel14_conf,
            'kernel30_est': kernel30_est,
            'kernel30_conf': kernel30_conf,
        })

    return pd.DataFrame(rows).set_index('index').reindex(query.index)


def build_temporal_tables(train, sub, hidden_mask, actual_mask, target):
    history_proxy = train.loc[~hidden_mask].copy()
    query_proxy = train.loc[hidden_mask, ['subject_id', 'sleep_date']].copy()
    query_actual = sub.loc[actual_mask, ['subject_id', 'sleep_date']].copy()
    proxy_features = build_temporal_features(history_proxy, query_proxy, target)
    actual_features = build_temporal_features(train, query_actual, target)
    return proxy_features, actual_features


def anchor_uncertainty(values, power):
    unc = 1.0 - 2.0 * np.abs(np.asarray(values, dtype=float) - 0.5)
    return np.clip(unc, 0.0, 1.0) ** power


def apply_smoothing(
    keys,
    anchor,
    temporal_by_target,
    mask,
    target_weights,
    mode,
    uncertainty_power,
    max_row_weight,
):
    out = keys.copy()
    movement = {}
    for target in TARGETS:
        out[target] = clip_prob(anchor[target])
        base_weight = float(target_weights.get(target, 0.0))
        if base_weight <= 0:
            movement[target] = {'mean_row_weight': 0.0, 'changed_rows': 0}
            continue

        temporal = temporal_by_target[target]
        est_col = f'{mode}_est'
        conf_col = f'{mode}_conf'
        if est_col not in temporal or conf_col not in temporal:
            raise ValueError(f'Unknown temporal mode: {mode}')

        anchor_values = anchor.loc[mask, target].to_numpy(dtype=float)
        estimates = temporal.loc[mask[mask].index, est_col].to_numpy(dtype=float)
        confidence = np.nan_to_num(temporal.loc[mask[mask].index, conf_col].to_numpy(dtype=float), nan=0.0)
        row_weight = base_weight * confidence * anchor_uncertainty(anchor_values, uncertainty_power)
        row_weight = np.clip(row_weight, 0.0, max_row_weight)
        adjusted = clip_prob((1.0 - row_weight) * anchor_values + row_weight * estimates)
        out.loc[mask, target] = adjusted
        movement[target] = {
            'mean_row_weight': float(np.mean(row_weight)) if len(row_weight) else 0.0,
            'max_row_weight': float(np.max(row_weight)) if len(row_weight) else 0.0,
            'changed_rows': int((row_weight > 1e-8).sum()),
        }
    return out, movement


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


def save_candidate(
    name,
    train,
    keys,
    anchor_oof,
    anchor_sub,
    proxy_temporal,
    actual_temporal,
    train_mask,
    sub_mask,
    eval_masks,
    target_weights,
    mode,
    uncertainty_power,
    max_row_weight,
):
    oof, oof_movement = apply_smoothing(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        anchor_oof,
        proxy_temporal,
        train_mask,
        target_weights,
        mode,
        uncertainty_power,
        max_row_weight,
    )
    submission, sub_movement = apply_smoothing(
        keys,
        anchor_sub,
        actual_temporal,
        sub_mask,
        target_weights,
        mode,
        uncertainty_power,
        max_row_weight,
    )

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
    for split in ['all_interior', 'simple_interior', 'fragmented_interior', 'tail', 'role_hybrid']:
        if split in metrics:
            print(f'  {split}: {metrics[split]["loss"]:.6f}')
    print(f'  weights={target_weights} mode={mode} uncertainty_power={uncertainty_power} max_row_weight={max_row_weight}')
    print(f'  oof_movement={oof_movement}')
    print(f'  sub_movement={sub_movement}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')

    return {
        'name': name,
        'mode': mode,
        'target_weights': target_weights,
        'uncertainty_power': uncertainty_power,
        'max_row_weight': max_row_weight,
        'metrics': metrics,
        'oof_movement': oof_movement,
        'sub_movement': sub_movement,
        'distribution': dist,
        'submission': str(sub_path),
        'oof_path': str(oof_path),
    }


def select_targets(anchor_oof, candidate_oof, train, simple_mask, fragmented_mask, all_mask, tolerance=0.0007):
    selected = {}
    diagnostics = {}
    for target in TARGETS:
        base_all = log_loss(train.loc[all_mask, target], np.clip(anchor_oof.loc[all_mask, target], 1e-7, 1 - 1e-7))
        cand_all = log_loss(train.loc[all_mask, target], np.clip(candidate_oof.loc[all_mask, target], 1e-7, 1 - 1e-7))
        base_simple = log_loss(train.loc[simple_mask, target], np.clip(anchor_oof.loc[simple_mask, target], 1e-7, 1 - 1e-7))
        cand_simple = log_loss(train.loc[simple_mask, target], np.clip(candidate_oof.loc[simple_mask, target], 1e-7, 1 - 1e-7))
        base_frag = log_loss(train.loc[fragmented_mask, target], np.clip(anchor_oof.loc[fragmented_mask, target], 1e-7, 1 - 1e-7))
        cand_frag = log_loss(train.loc[fragmented_mask, target], np.clip(candidate_oof.loc[fragmented_mask, target], 1e-7, 1 - 1e-7))
        diagnostics[target] = {
            'all_delta': float(cand_all - base_all),
            'simple_delta': float(cand_simple - base_simple),
            'fragmented_delta': float(cand_frag - base_frag),
        }
        if cand_all < base_all and cand_simple <= base_simple + tolerance and cand_frag <= base_frag + tolerance:
            selected[target] = True
    return selected, diagnostics


def main():
    ensure_dirs()
    log_path = LOG_DIR / f'run_{EXP_TAG}.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG}...')
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_PATH)
    keys = sub[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    anchor_oof = load_frame(ANCHOR_OOF_PATH)
    anchor_sub = load_frame(ANCHOR_SUB_PATH)

    profiles = interior_eval.build_profiles(train, sub)
    simple_mask, fragmented_mask, all_interior_mask, tail_mask = build_disjoint_proxy_role_masks(train, profiles)
    hidden_mask = all_interior_mask | tail_mask
    actual_interior_mask = build_actual_interior_mask(train, sub)
    actual_tail_mask = pd.Series(~actual_interior_mask.to_numpy(), index=sub.index)

    eval_masks = {
        'all_interior': all_interior_mask,
        'simple_interior': simple_mask,
        'fragmented_interior': fragmented_mask,
        'tail': tail_mask,
        'role_hybrid': hidden_mask,
    }
    print(
        f'proxy rows: interior={int(all_interior_mask.sum())} '
        f'simple={int(simple_mask.sum())} fragmented={int(fragmented_mask.sum())} '
        f'tail={int(tail_mask.sum())} hidden={int(hidden_mask.sum())}'
    )
    print(
        f'actual rows: interior={int(actual_interior_mask.sum())} '
        f'tail={int(actual_tail_mask.sum())}'
    )

    print('[v45] building temporal tables...')
    proxy_temporal = {}
    actual_temporal = {}
    for target in TARGETS:
        proxy_temporal[target], actual_temporal[target] = build_temporal_tables(
            train,
            sub,
            hidden_mask,
            actual_interior_mask,
            target,
        )

    anchor_metrics = {
        split: {'loss': evaluate(train, anchor_oof, mask)[0]}
        for split, mask in eval_masks.items()
    }
    print('[v45] anchor metrics')
    for split, result in anchor_metrics.items():
        print(f'  {split}: {result["loss"]:.6f}')

    specs = [
        ('agree_w20_u1', 'agree', 0.20, 1.0, 0.12),
        ('agree_w35_u1', 'agree', 0.35, 1.0, 0.18),
        ('bracket_w08_u1', 'bracket', 0.08, 1.0, 0.08),
        ('bracket_w12_u15', 'bracket', 0.12, 1.5, 0.10),
        ('kernel14_w06_u1', 'kernel14', 0.06, 1.0, 0.06),
        ('kernel14_w10_u15', 'kernel14', 0.10, 1.5, 0.08),
        ('kernel30_w08_u15', 'kernel30', 0.08, 1.5, 0.07),
    ]

    summaries = []
    target_select_specs = []
    for tag, mode, base_weight, uncertainty_power, max_row_weight in specs:
        weights = {target: base_weight for target in TARGETS}
        name = f'{EXP_TAG}_interior_{tag}'
        summary = save_candidate(
            name,
            train,
            keys,
            anchor_oof,
            anchor_sub,
            proxy_temporal,
            actual_temporal,
            all_interior_mask,
            actual_interior_mask,
            eval_masks,
            weights,
            mode,
            uncertainty_power,
            max_row_weight,
        )
        summaries.append(summary)

        cand_oof = load_frame(summary['oof_path'])
        selected_bool, diagnostics = select_targets(
            anchor_oof, cand_oof, train, simple_mask, fragmented_mask, all_interior_mask)
        selected_weights = {
            target: base_weight
            for target, keep in selected_bool.items()
            if keep
        }
        target_select_specs.append({
            'tag': tag,
            'mode': mode,
            'base_weight': base_weight,
            'uncertainty_power': uncertainty_power,
            'max_row_weight': max_row_weight,
            'selected_weights': selected_weights,
            'diagnostics': diagnostics,
        })
        if selected_weights:
            summaries.append(save_candidate(
                f'{EXP_TAG}_interior_{tag}_targetselect',
                train,
                keys,
                anchor_oof,
                anchor_sub,
                proxy_temporal,
                actual_temporal,
                all_interior_mask,
                actual_interior_mask,
                eval_masks,
                selected_weights,
                mode,
                uncertainty_power,
                max_row_weight,
            ))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'anchor': str(ANCHOR_SUB_PATH),
        'proxy_rows': {
            'all_interior': int(all_interior_mask.sum()),
            'simple_interior': int(simple_mask.sum()),
            'fragmented_interior': int(fragmented_mask.sum()),
            'tail': int(tail_mask.sum()),
            'hidden': int(hidden_mask.sum()),
        },
        'actual_rows': {
            'interior': int(actual_interior_mask.sum()),
            'tail': int(actual_tail_mask.sum()),
        },
        'anchor_metrics': anchor_metrics,
        'target_select_specs': target_select_specs,
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

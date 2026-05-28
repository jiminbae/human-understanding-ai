# v46: row-wise adaptive S4 temporal smoothing.
#   - v45 showed that S4-only temporal agreement is a public-valid correction.
#   - The global strength peaked around w65 and worsened by w75, so this version
#     keeps the average correction near w60-w65 while redistributing strength:
#     confident/uncertain-anchor rows are pushed harder, risky rows are guarded.
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from jimin.analysis import pseudo_public_interior_profile_eval as interior_eval
from jimin.models import baseline_v45_uncertainty_temporal_smoothing as v45


TARGET = 'S4'
TARGETS = v45.TARGETS
EXP_TAG = os.environ.get('V46_EXP_TAG', 'v46_adaptive_s4_temporal_smoothing')

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'


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


def base_output(keys, anchor):
    out = keys.copy()
    for target in TARGETS:
        out[target] = v45.clip_prob(anchor[target])
    return out


def build_row_frame(anchor, temporal_s4, mask):
    idx = mask[mask].index
    anchor_values = anchor.loc[idx, TARGET].to_numpy(dtype=float)
    estimates = temporal_s4.loc[idx, 'agree_est'].to_numpy(dtype=float)
    confidence = np.nan_to_num(
        temporal_s4.loc[idx, 'agree_conf'].to_numpy(dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    uncertainty = v45.anchor_uncertainty(anchor_values, 1.0)
    prev_gap = np.nan_to_num(
        temporal_s4.loc[idx, 'prev_gap'].to_numpy(dtype=float),
        nan=np.inf,
        posinf=np.inf,
    )
    next_gap = np.nan_to_num(
        temporal_s4.loc[idx, 'next_gap'].to_numpy(dtype=float),
        nan=np.inf,
        posinf=np.inf,
    )
    max_gap = np.maximum(prev_gap, next_gap)
    abs_delta = np.abs(estimates - anchor_values)
    return pd.DataFrame({
        'anchor': anchor_values,
        'estimate': estimates,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'abs_delta': abs_delta,
        'prev_gap': prev_gap,
        'next_gap': next_gap,
        'max_gap': max_gap,
    }, index=idx)


def global_rule(row_frame, weight):
    base = np.full(len(row_frame), weight, dtype=float)
    cap = np.full(len(row_frame), weight * 0.60, dtype=float)
    return base, cap


def rule_balanced_conf_delta(row_frame):
    conf = row_frame['confidence'].to_numpy(dtype=float)
    unc = row_frame['uncertainty'].to_numpy(dtype=float)
    delta = row_frame['abs_delta'].to_numpy(dtype=float)

    base = np.full(len(row_frame), 0.50, dtype=float)
    base[(conf >= 0.66) & (unc >= 0.50)] = 0.60
    base[(conf >= 0.74) & (unc >= 0.58)] = 0.72
    base[(conf >= 0.84) & (unc >= 0.70) & (delta <= 0.52)] = 0.88
    base[delta >= 0.56] = np.minimum(base[delta >= 0.56], 0.52)
    cap = 0.60 * base
    return base, cap


def rule_highconf_push_guarded(row_frame):
    conf = row_frame['confidence'].to_numpy(dtype=float)
    unc = row_frame['uncertainty'].to_numpy(dtype=float)
    delta = row_frame['abs_delta'].to_numpy(dtype=float)
    max_gap = row_frame['max_gap'].to_numpy(dtype=float)

    base = np.full(len(row_frame), 0.45, dtype=float)
    base[(conf >= 0.63) & (unc >= 0.42)] = 0.56
    base[(conf >= 0.74) & (unc >= 0.55)] = 0.74
    base[(conf >= 0.84) & (unc >= 0.62) & (max_gap <= 9.0)] = 0.95
    base[(delta >= 0.54) | (max_gap >= 15.0)] = np.minimum(
        base[(delta >= 0.54) | (max_gap >= 15.0)],
        0.55,
    )
    cap = 0.58 * base
    return base, cap


def rule_uncertainty_guarded(row_frame):
    conf = row_frame['confidence'].to_numpy(dtype=float)
    unc = row_frame['uncertainty'].to_numpy(dtype=float)
    delta = row_frame['abs_delta'].to_numpy(dtype=float)

    base = np.full(len(row_frame), 0.44, dtype=float)
    base[unc >= 0.50] = 0.56
    base[unc >= 0.68] = 0.70
    base[unc >= 0.86] = 0.90
    base[conf < 0.68] *= 0.82
    base[(conf >= 0.84) & (unc >= 0.80) & (delta <= 0.50)] = 1.00
    base[delta >= 0.57] = np.minimum(base[delta >= 0.57], 0.48)
    cap = 0.56 * base
    return base, cap


def apply_adaptive_s4(keys, anchor, temporal_s4, mask, rule_name, rule_func):
    out = base_output(keys, anchor)
    row_frame = build_row_frame(anchor, temporal_s4, mask)
    base_weight, cap = rule_func(row_frame)

    conf = row_frame['confidence'].to_numpy(dtype=float)
    unc = row_frame['uncertainty'].to_numpy(dtype=float)
    anchor_values = row_frame['anchor'].to_numpy(dtype=float)
    estimates = row_frame['estimate'].to_numpy(dtype=float)
    row_weight = np.clip(base_weight * conf * unc, 0.0, cap)
    adjusted = v45.clip_prob((1.0 - row_weight) * anchor_values + row_weight * estimates)
    out.loc[row_frame.index, TARGET] = adjusted

    active = row_weight > 1e-8
    movement = {
        'rule': rule_name,
        'rows': int(len(row_frame)),
        'changed_rows': int(active.sum()),
        'mean_base_weight_all': float(np.mean(base_weight)) if len(base_weight) else 0.0,
        'mean_base_weight_active': float(np.mean(base_weight[active])) if active.any() else 0.0,
        'mean_row_weight_all': float(np.mean(row_weight)) if len(row_weight) else 0.0,
        'mean_row_weight_active': float(np.mean(row_weight[active])) if active.any() else 0.0,
        'max_row_weight': float(np.max(row_weight)) if len(row_weight) else 0.0,
        'mean_confidence_active': float(np.mean(conf[active])) if active.any() else 0.0,
        'mean_uncertainty_active': float(np.mean(unc[active])) if active.any() else 0.0,
        'base_weight_quantiles_active': (
            np.quantile(base_weight[active], [0.0, 0.25, 0.5, 0.75, 1.0]).round(6).tolist()
            if active.any() else []
        ),
        'row_weight_quantiles_active': (
            np.quantile(row_weight[active], [0.0, 0.25, 0.5, 0.75, 1.0]).round(6).tolist()
            if active.any() else []
        ),
    }
    return out, movement


def save_candidate(
    name,
    train,
    keys,
    anchor_oof,
    anchor_sub,
    proxy_temporal_s4,
    actual_temporal_s4,
    train_mask,
    sub_mask,
    eval_masks,
    rule_func,
):
    oof, oof_movement = apply_adaptive_s4(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        anchor_oof,
        proxy_temporal_s4,
        train_mask,
        name,
        rule_func,
    )
    submission, sub_movement = apply_adaptive_s4(
        keys,
        anchor_sub,
        actual_temporal_s4,
        sub_mask,
        name,
        rule_func,
    )

    metrics = {}
    for mask_name, mask in eval_masks.items():
        total, per_target = v45.evaluate(train, oof, mask)
        metrics[mask_name] = {'loss': total, 'per_target': per_target}

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    dist = v45.describe_vs_anchor(submission, anchor_sub)
    print(f'\n{name}')
    for split in ['all_interior', 'simple_interior', 'fragmented_interior', 'tail', 'role_hybrid']:
        print(f'  {split}: {metrics[split]["loss"]:.6f}')
    print(f'  oof_movement={oof_movement}')
    print(f'  sub_movement={sub_movement}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')

    return {
        'name': name,
        'metrics': metrics,
        'oof_movement': oof_movement,
        'sub_movement': sub_movement,
        'distribution': dist,
        'submission': str(sub_path),
        'oof_path': str(oof_path),
    }


def main():
    ensure_dirs()
    log_path = LOG_DIR / f'run_{EXP_TAG}.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG}...')
    train = v45.load_frame(v45.TRAIN_PATH)
    sub = v45.load_frame(v45.SUB_PATH)
    keys = sub[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    anchor_oof = v45.load_frame(v45.ANCHOR_OOF_PATH)
    anchor_sub = v45.load_frame(v45.ANCHOR_SUB_PATH)

    profiles = interior_eval.build_profiles(train, sub)
    simple_mask, fragmented_mask, all_interior_mask, tail_mask = v45.build_disjoint_proxy_role_masks(train, profiles)
    hidden_mask = all_interior_mask | tail_mask
    actual_interior_mask = v45.build_actual_interior_mask(train, sub)
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

    print('[v46] building S4 temporal tables...')
    proxy_temporal_s4, actual_temporal_s4 = v45.build_temporal_tables(
        train,
        sub,
        hidden_mask,
        actual_interior_mask,
        TARGET,
    )

    candidates = [
        ('global_w60', lambda rows: global_rule(rows, 0.60)),
        ('global_w62', lambda rows: global_rule(rows, 0.62)),
        ('balanced_conf_delta', rule_balanced_conf_delta),
        ('highconf_push_guarded', rule_highconf_push_guarded),
        ('uncertainty_guarded', rule_uncertainty_guarded),
    ]

    summaries = []
    for suffix, rule_func in candidates:
        name = f'{EXP_TAG}_{suffix}'
        summaries.append(save_candidate(
            name,
            train,
            keys,
            anchor_oof,
            anchor_sub,
            proxy_temporal_s4,
            actual_temporal_s4,
            all_interior_mask,
            actual_interior_mask,
            eval_masks,
            rule_func,
        ))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'anchor': str(v45.ANCHOR_SUB_PATH),
        'known_public_best_before_v46': {
            'submission': 'submission_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv',
            'score': 0.5831903668,
        },
        'latest_public_probe': {
            'submission': 'submission_v45_uncertainty_temporal_smoothing_interior_agree_w75_u1_s4only.csv',
            'score': 0.5832678176,
            'interpretation': 'global strength likely peaked before w75; adaptive row-wise correction should keep average near w60-w65.',
        },
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
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

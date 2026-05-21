# v37: disagreement-gated selective reuse of the v33 winning signals.
#   - v32 remains the public-stable anchor.
#   - v33 signals are reused only where OOF disagreement suggested they are reliable:
#       Q1: use Pass1 on high-disagreement rows.
#       Q3: use Pass2 on low-disagreement rows.
#       S4: use Pass2 on mid-disagreement rows.
#   - Gate thresholds are fixed from OOF disagreement quantiles and then reused on test.
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

EXP_TAG = os.environ.get('V37_EXP_TAG', 'v37_disagreement_gate')

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


def evaluate(train, pred):
    per_target = {
        target: log_loss(train[target].values, np.clip(pred[target].values, 1e-7, 1 - 1e-7))
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def build_tail_mask(train, frac=0.2):
    mask = np.zeros(len(train), dtype=bool)
    ordered = train.reset_index().sort_values(['subject_id', 'sleep_date'])
    for _, grp in ordered.groupby('subject_id'):
        n_tail = max(1, int(np.ceil(len(grp) * frac)))
        mask[grp.tail(n_tail)['index'].to_numpy()] = True
    return mask


def evaluate_subset(train, pred, mask):
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
        'corr_vs_v32': float(np.corrcoef(ref, arr)[0, 1]),
        'mad_vs_v32': float(np.mean(np.abs(ref - arr))),
        'max_abs_vs_v32': float(np.max(np.abs(ref - arr))),
        'means': {target: float(submission[target].mean()) for target in TARGETS},
    }


def fit_gate_thresholds(anchor_oof, pass1_oof, pass2_oof):
    q1_diff = np.abs(pass1_oof['Q1'].to_numpy() - anchor_oof['Q1'].to_numpy())
    q3_diff = np.abs(pass2_oof['Q3'].to_numpy() - anchor_oof['Q3'].to_numpy())
    s4_diff = np.abs(pass2_oof['S4'].to_numpy() - anchor_oof['S4'].to_numpy())
    return {
        'Q1': {
            'mode': 'high',
            'candidate': 'pass1',
            'quantiles': [0.50],
            'thresholds': [float(np.quantile(q1_diff, 0.50))],
        },
        'Q3': {
            'mode': 'low',
            'candidate': 'pass2',
            'quantiles': [0.60],
            'thresholds': [float(np.quantile(q3_diff, 0.60))],
        },
        'S4': {
            'mode': 'band',
            'candidate': 'pass2',
            'quantiles': [0.30, 0.70],
            'thresholds': [
                float(np.quantile(s4_diff, 0.30)),
                float(np.quantile(s4_diff, 0.70)),
            ],
        },
    }


def candidate_frame(source_name, pass1, pass2):
    if source_name == 'pass1':
        return pass1
    if source_name == 'pass2':
        return pass2
    raise ValueError(f'Unknown candidate source: {source_name}')


def gate_mask(anchor, candidate, spec, target):
    diff = np.abs(candidate[target].to_numpy() - anchor[target].to_numpy())
    mode = spec['mode']
    thresholds = spec['thresholds']
    if mode == 'high':
        return diff >= thresholds[0]
    if mode == 'low':
        return diff <= thresholds[0]
    if mode == 'band':
        return (diff >= thresholds[0]) & (diff <= thresholds[1])
    raise ValueError(f'Unknown gate mode: {mode}')


def apply_sources(keys, anchor, pass1, pass2, specs):
    out = keys.copy()
    usage = {}
    for target in TARGETS:
        spec = specs.get(target)
        if spec is None:
            out[target] = clip_prob(anchor[target])
            usage[target] = 0.0
            continue

        if spec['type'] == 'full':
            candidate = candidate_frame(spec['candidate'], pass1, pass2)
            out[target] = clip_prob(candidate[target])
            usage[target] = 1.0
            continue

        if spec['type'] == 'gate':
            candidate = candidate_frame(spec['candidate'], pass1, pass2)
            use_candidate = gate_mask(anchor, candidate, spec, target)
            out[target] = clip_prob(np.where(use_candidate, candidate[target], anchor[target]))
            usage[target] = float(np.mean(use_candidate))
            continue

        raise ValueError(f'Unknown spec type for {target}: {spec["type"]}')
    return out, usage


def save_candidate(
    name,
    train,
    keys,
    anchor_oof,
    pass1_oof,
    pass2_oof,
    anchor_sub,
    pass1_sub,
    pass2_sub,
    specs,
    tail_mask,
):
    oof, oof_usage = apply_sources(
        train[['subject_id', 'sleep_date', 'lifelog_date']],
        anchor_oof,
        pass1_oof,
        pass2_oof,
        specs,
    )
    submission, sub_usage = apply_sources(
        keys,
        anchor_sub,
        pass1_sub,
        pass2_sub,
        specs,
    )
    oof_total, per_target = evaluate(train, oof)
    tail_total, tail_per_target = evaluate_subset(train, oof, tail_mask)

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)

    dist = describe_vs_anchor(submission, anchor_sub)
    print(f'\n{name}: OOF={oof_total:.6f}  pseudo_tail={tail_total:.6f}')
    print(f'  per_target={per_target}')
    print(f'  tail_per_target={tail_per_target}')
    print(f'  oof_usage={oof_usage}')
    print(f'  sub_usage={sub_usage}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')
    return {
        'name': name,
        'specs': specs,
        'oof': oof_total,
        'pseudo_tail_oof': tail_total,
        'per_target': per_target,
        'pseudo_tail_per_target': tail_per_target,
        'oof_usage': oof_usage,
        'submission_usage': sub_usage,
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

    gate_specs = fit_gate_thresholds(anchor_oof, pass1_oof, pass2_oof)
    tail_mask = build_tail_mask(train)
    print(f'Gate specs: {gate_specs}')

    q1_gate = {'type': 'gate', **gate_specs['Q1']}
    q3_gate = {'type': 'gate', **gate_specs['Q3']}
    s4_gate = {'type': 'gate', **gate_specs['S4']}
    q1_full = {'type': 'full', 'candidate': 'pass1'}
    q3_full = {'type': 'full', 'candidate': 'pass2'}
    s4_full = {'type': 'full', 'candidate': 'pass2'}

    candidates = {
        # Current public winner reproduced for direct comparison.
        f'{EXP_TAG}_winner_repro': {
            'Q1': q1_full,
            'Q3': q3_full,
            'S4': s4_full,
        },
        # Main recommendation: keep the public-proven Q3 replacement, gate Q1/S4.
        f'{EXP_TAG}_q1gate_q3p2_s4gate': {
            'Q1': q1_gate,
            'Q3': q3_full,
            'S4': s4_gate,
        },
        # Most aggressive OOF winner: gate all three targets.
        f'{EXP_TAG}_allgate': {
            'Q1': q1_gate,
            'Q3': q3_gate,
            'S4': s4_gate,
        },
        # Cleaner public probe: only add the strongest new gate on top of the winner.
        f'{EXP_TAG}_q1p1_q3p2_s4gate': {
            'Q1': q1_full,
            'Q3': q3_full,
            'S4': s4_gate,
        },
        # Separate gate diagnostics.
        f'{EXP_TAG}_q1gate_only': {
            'Q1': q1_gate,
        },
        f'{EXP_TAG}_q3gate_only': {
            'Q3': q3_gate,
        },
        f'{EXP_TAG}_s4gate_only': {
            'S4': s4_gate,
        },
    }

    summaries = []
    for name, specs in candidates.items():
        summaries.append(save_candidate(
            name,
            train,
            keys,
            anchor_oof,
            pass1_oof,
            pass2_oof,
            anchor_sub,
            pass1_sub,
            pass2_sub,
            specs,
            tail_mask,
        ))

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'anchor': 'v32_target_specialized_bidir_s2w0p675',
        'pass1': 'v33_long_history_cross_target_pass1',
        'pass2': 'v33_long_history_cross_target_pass2',
        'gate_specs': gate_specs,
        'candidates': summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

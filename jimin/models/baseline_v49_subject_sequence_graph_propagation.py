# v49: subject sequence graph / label-propagation model.
#
# v48 proved that the v47 raw-sensor model should be trusted very differently
# by target. v49 adds a new information source: the subject-level time graph.
# It predicts hidden rows from nearby known train labels, then blends that
# sequence prediction with the best available v48 policy submission.
#
# This script is intentionally light. It does not train a large model; it builds
# test-like pseudo-hidden validation blocks inside train, uses the same graph
# rule for train/test, and writes several candidate submissions.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

RAW_OOF = OOF_DIR / 'oof_v47_hourgrid_subject_state_residual_raw.csv'
RAW_SUB = SUB_DIR / 'submission_v47_hourgrid_subject_state_residual_raw.csv'
ANCHOR_OOF = OOF_DIR / 'oof_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
ANCHOR_SUB = SUB_DIR / 'submission_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'

DEFAULT_BASE_TAGS = [
    # Future v48 follow-ups can be dropped in without changing this script.
    'v48_target_delta_scaled_avg310_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg270_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg250_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg230_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg190_q2cap115_q3s3guard',
    'v48_target_delta_scaled_avg145_q3s3guard',
    'v48_target_delta_scaled_avg115_q3s3guard',
    'v48_target_delta_scaled_avg085_cap12',
    'v48_target_delta_scaled_avg07',
]

# Pseudo-hidden windows. These are deliberately block-shaped, not random KFold.
PSEUDO_BLOCK_LENGTHS = [1, 2, 3, 5, 8]


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['sleep_date', 'lifelog_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.reset_index(drop=True)


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


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


def choose_base_prediction():
    env_tag = os.environ.get('V49_BASE_TAG')
    tags = [env_tag] if env_tag else []
    tags.extend(DEFAULT_BASE_TAGS)

    for tag in tags:
        if not tag:
            continue
        oof_path = OOF_DIR / f'oof_{tag}.csv'
        sub_path = SUB_DIR / f'submission_{tag}.csv'
        if oof_path.exists() and sub_path.exists():
            return tag, load_frame(oof_path), load_frame(sub_path), str(oof_path), str(sub_path)

    if ANCHOR_OOF.exists() and ANCHOR_SUB.exists():
        return 'v45_anchor_w65', load_frame(ANCHOR_OOF), load_frame(ANCHOR_SUB), str(ANCHOR_OOF), str(ANCHOR_SUB)

    raise FileNotFoundError('No usable base prediction found for v49.')


def subject_target_priors(train):
    global_mean = {target: float(train[target].mean()) for target in TARGETS}
    subject_means = {
        target: train.groupby('subject_id')[target].mean().to_dict()
        for target in TARGETS
    }
    return global_mean, subject_means


def smoothed_subject_mean(subject_id, target, global_mean, subject_means):
    # Ten subjects only: keep subject priors useful, but not absolute.
    subj = float(subject_means[target].get(subject_id, global_mean[target]))
    return 0.78 * subj + 0.22 * global_mean[target]


def label_to_prob(label, prior, confidence=0.84):
    # Convert hard train labels into calibrated probabilities.
    label_prob = 0.86 if float(label) >= 0.5 else 0.14
    return confidence * label_prob + (1.0 - confidence) * prior


def find_neighbor_positions(known_positions, pos):
    prev_positions = known_positions[known_positions < pos]
    next_positions = known_positions[known_positions > pos]
    prev_pos = int(prev_positions[-1]) if len(prev_positions) else None
    next_pos = int(next_positions[0]) if len(next_positions) else None
    return prev_pos, next_pos


def recent_known_mean(seq, known_positions, target, pos, side, max_count=3):
    if side == 'prev':
        positions = known_positions[known_positions < pos][-max_count:]
    else:
        positions = known_positions[known_positions > pos][:max_count]
    if len(positions) == 0:
        return np.nan
    return float(seq.loc[positions, target].mean())


def graph_predict_for_row(seq, known_positions, row_pos, target, global_mean, subject_means):
    row = seq.loc[row_pos]
    subject_id = row['subject_id']
    prior = smoothed_subject_mean(subject_id, target, global_mean, subject_means)
    prev_pos, next_pos = find_neighbor_positions(known_positions, row_pos)

    pieces = []
    weights = []
    meta = {
        'has_prev': prev_pos is not None,
        'has_next': next_pos is not None,
        'is_interior': prev_pos is not None and next_pos is not None,
        'prev_next_agree': False,
        'dist_prev': None,
        'dist_next': None,
    }

    if prev_pos is not None:
        dist_prev = max(1, int((row['sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days))
        meta['dist_prev'] = dist_prev
        p_prev = label_to_prob(seq.loc[prev_pos, target], prior, confidence=0.86)
        pieces.append(p_prev)
        weights.append(1.0 / np.sqrt(dist_prev))
    if next_pos is not None:
        dist_next = max(1, int((seq.loc[next_pos, 'sleep_date'] - row['sleep_date']).days))
        meta['dist_next'] = dist_next
        p_next = label_to_prob(seq.loc[next_pos, target], prior, confidence=0.86)
        pieces.append(p_next)
        weights.append(1.0 / np.sqrt(dist_next))

    if pieces:
        neighbor = float(np.average(pieces, weights=weights))
    else:
        neighbor = prior

    prev_recent = recent_known_mean(seq, known_positions, target, row_pos, 'prev')
    next_recent = recent_known_mean(seq, known_positions, target, row_pos, 'next')
    recent_parts = [value for value in [prev_recent, next_recent] if np.isfinite(value)]
    if recent_parts:
        recent_label_mean = float(np.mean(recent_parts))
        recent = label_to_prob(recent_label_mean, prior, confidence=0.58)
    else:
        recent = prior

    if prev_pos is not None and next_pos is not None:
        prev_label = float(seq.loc[prev_pos, target])
        next_label = float(seq.loc[next_pos, target])
        meta['prev_next_agree'] = abs(prev_label - next_label) < 1e-12
        span = max(1, int((seq.loc[next_pos, 'sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days))
        pos_frac = max(0.0, min(1.0, float((row['sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days) / span))
        linear_label = (1.0 - pos_frac) * prev_label + pos_frac * next_label
        linear = label_to_prob(linear_label, prior, confidence=0.66)
        if meta['prev_next_agree']:
            agree = label_to_prob(prev_label, prior, confidence=0.91)
            pred = 0.46 * agree + 0.30 * neighbor + 0.16 * linear + 0.08 * recent
            confidence = 0.78
        else:
            pred = 0.42 * neighbor + 0.30 * linear + 0.18 * recent + 0.10 * prior
            confidence = 0.58
    elif prev_pos is not None or next_pos is not None:
        pred = 0.55 * neighbor + 0.25 * recent + 0.20 * prior
        nearest_dist = meta['dist_prev'] if prev_pos is not None else meta['dist_next']
        confidence = float(max(0.28, min(0.58, 0.62 / np.sqrt(max(1, nearest_dist)))))
    else:
        pred = prior
        confidence = 0.18

    return float(np.clip(pred, 0.04, 0.96)), confidence, meta


def prepare_sequence_frame(train, sub):
    train_seq = train[KEYS + TARGETS].copy()
    train_seq['_split'] = 'train'
    train_seq['_orig_index'] = np.arange(len(train_seq))

    sub_seq = sub[KEYS].copy()
    for target in TARGETS:
        sub_seq[target] = np.nan
    sub_seq['_split'] = 'test'
    sub_seq['_orig_index'] = np.arange(len(sub_seq))

    seq = pd.concat([train_seq, sub_seq], ignore_index=True)
    seq = seq.sort_values(['subject_id', 'sleep_date', 'lifelog_date', '_split']).reset_index(drop=True)
    return seq


def build_graph_submission(train, sub):
    global_mean, subject_means = subject_target_priors(train)
    seq = prepare_sequence_frame(train, sub)
    graph_sub = make_keys(sub)
    conf_sub = make_keys(sub)
    role_rows = []

    for subject_id, grp in seq.groupby('subject_id', sort=False):
        grp = grp.reset_index(drop=True)
        known_positions = grp.index[grp['_split'] == 'train'].to_numpy(dtype=int)
        for pos, row in grp.loc[grp['_split'] == 'test'].iterrows():
            out_idx = int(row['_orig_index'])
            role_record = {
                'subject_id': subject_id,
                'test_index': out_idx,
                'sleep_date': str(row['sleep_date'].date()),
            }
            for target in TARGETS:
                pred, conf, meta = graph_predict_for_row(
                    grp,
                    known_positions,
                    int(pos),
                    target,
                    global_mean,
                    subject_means,
                )
                graph_sub.loc[out_idx, target] = pred
                conf_sub.loc[out_idx, target] = conf
                if target == 'Q1':
                    role_record.update({
                        'is_interior': bool(meta['is_interior']),
                        'prev_next_agree': bool(meta['prev_next_agree']),
                        'has_prev': bool(meta['has_prev']),
                        'has_next': bool(meta['has_next']),
                        'dist_prev': meta['dist_prev'],
                        'dist_next': meta['dist_next'],
                    })
            role_rows.append(role_record)

    role_df = pd.DataFrame(role_rows).sort_values('test_index').reset_index(drop=True)
    return graph_sub, conf_sub, role_df


def build_graph_pseudo_oof(train):
    global_mean, subject_means = subject_target_priors(train)
    pred_sums = pd.DataFrame(0.0, index=train.index, columns=TARGETS)
    conf_sums = pd.DataFrame(0.0, index=train.index, columns=TARGETS)
    counts = pd.Series(0, index=train.index, dtype=int)

    for subject_id, grp in train.sort_values(['subject_id', 'sleep_date']).groupby('subject_id', sort=False):
        grp = grp.reset_index(drop=False).rename(columns={'index': '_train_index'})
        n = len(grp)
        for length in PSEUDO_BLOCK_LENGTHS:
            if length > n:
                continue
            for start in range(0, n - length + 1):
                hidden_positions = set(range(start, start + length))
                known_positions = np.array([pos for pos in range(n) if pos not in hidden_positions], dtype=int)
                if len(known_positions) == 0:
                    continue
                for pos in hidden_positions:
                    train_idx = int(grp.loc[pos, '_train_index'])
                    for target in TARGETS:
                        pred, conf, _ = graph_predict_for_row(
                            grp,
                            known_positions,
                            pos,
                            target,
                            global_mean,
                            subject_means,
                        )
                        pred_sums.loc[train_idx, target] += pred
                        conf_sums.loc[train_idx, target] += conf
                    counts.loc[train_idx] += 1

    graph_oof = make_keys(train)
    conf_oof = make_keys(train)
    global_mean, subject_means = subject_target_priors(train)
    for target in TARGETS:
        fallback = train['subject_id'].map(
            lambda sid: smoothed_subject_mean(sid, target, global_mean, subject_means)
        ).to_numpy(dtype=float)
        cnt = counts.to_numpy(dtype=float)
        graph_oof[target] = np.where(cnt > 0, pred_sums[target].to_numpy(dtype=float) / np.maximum(cnt, 1), fallback)
        conf_oof[target] = np.where(cnt > 0, conf_sums[target].to_numpy(dtype=float) / np.maximum(cnt, 1), 0.18)
        graph_oof[target] = clip_prob(graph_oof[target])
        conf_oof[target] = np.clip(conf_oof[target].to_numpy(dtype=float), 0.0, 1.0)
    return graph_oof, conf_oof


def blend_frames(base, other, weights):
    out = base.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        if weight <= 0:
            continue
        out[target] = clip_prob((1.0 - weight) * base[target] + weight * other[target])
    return out


def confidence_blend(base, graph, conf, target_high_weights, low_scale=0.35):
    out = base.copy()
    diagnostics = {}
    for target in TARGETS:
        high = float(target_high_weights.get(target, 0.0))
        if high <= 0:
            continue
        c = conf[target].to_numpy(dtype=float)
        row_weight = high * (low_scale + (1.0 - low_scale) * c)
        row_weight = np.clip(row_weight, 0.0, high)
        out[target] = clip_prob((1.0 - row_weight) * base[target] + row_weight * graph[target])
        diagnostics[target] = {
            'mean_weight': float(row_weight.mean()),
            'max_weight': float(row_weight.max()),
        }
    return out, diagnostics


def describe_vs_base(pred, base):
    pred_arr = pred[TARGETS].to_numpy(dtype=float)
    base_arr = base[TARGETS].to_numpy(dtype=float)
    diff = pred_arr - base_arr
    return {
        'corr_vs_base': float(np.corrcoef(pred_arr.ravel(), base_arr.ravel())[0, 1]),
        'mad_vs_base': float(np.mean(np.abs(diff))),
        'max_abs_vs_base': float(np.max(np.abs(diff))),
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
        'oof_loss': oof_loss,
        'oof_per_target': per_target,
        'distribution_vs_base': describe_vs_base(submission, base_sub),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
    }


def main():
    ensure_dirs()
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_SAMPLE_PATH)
    base_tag, base_oof, base_sub, base_oof_path, base_sub_path = choose_base_prediction()
    raw_oof = load_frame(RAW_OOF)
    raw_sub = load_frame(RAW_SUB)

    print(f'[v49] base={base_tag}')
    graph_sub, graph_conf_sub, role_df = build_graph_submission(train, sub)
    graph_oof, graph_conf_oof = build_graph_pseudo_oof(train)

    graph_oof_path = OOF_DIR / 'oof_v49_sequence_graph_only.csv'
    graph_sub_path = SUB_DIR / 'submission_v49_sequence_graph_only.csv'
    graph_conf_oof_path = OOF_DIR / 'oof_v49_sequence_graph_confidence.csv'
    graph_conf_sub_path = SUB_DIR / 'submission_v49_sequence_graph_confidence.csv'
    role_path = SUMMARY_DIR / 'v49_test_sequence_roles.csv'
    graph_oof.to_csv(graph_oof_path, index=False)
    graph_sub.to_csv(graph_sub_path, index=False)
    graph_conf_oof.to_csv(graph_conf_oof_path, index=False)
    graph_conf_sub.to_csv(graph_conf_sub_path, index=False)
    role_df.to_csv(role_path, index=False)

    graph_loss, graph_per_target = evaluate_frame(train, graph_oof)
    base_loss, base_per_target = evaluate_frame(train, base_oof)
    raw_loss, raw_per_target = evaluate_frame(train, raw_oof)

    candidates = []
    candidates.append({
        'name': 'v49_sequence_graph_only',
        'policy': {'type': 'graph_only'},
        'oof_loss': graph_loss,
        'oof_per_target': graph_per_target,
        'distribution_vs_base': describe_vs_base(graph_sub, base_sub),
        'oof_path': str(graph_oof_path),
        'submission': str(graph_sub_path),
    })

    blend_specs = {
        'v49_base_graph_safe': {
            'Q1': 0.08, 'Q2': 0.06, 'Q3': 0.05,
            'S1': 0.10, 'S2': 0.10, 'S3': 0.05, 'S4': 0.08,
        },
        'v49_base_graph_mid': {
            'Q1': 0.14, 'Q2': 0.09, 'Q3': 0.06,
            'S1': 0.16, 'S2': 0.15, 'S3': 0.06, 'S4': 0.12,
        },
        'v49_base_graph_q1s1s2strong': {
            'Q1': 0.22, 'Q2': 0.10, 'Q3': 0.04,
            'S1': 0.25, 'S2': 0.23, 'S3': 0.05, 'S4': 0.10,
        },
        'v49_base_graph_s4probe': {
            'Q1': 0.12, 'Q2': 0.08, 'Q3': 0.04,
            'S1': 0.14, 'S2': 0.14, 'S3': 0.04, 'S4': 0.22,
        },
    }
    for name, weights in blend_specs.items():
        oof_pred = blend_frames(base_oof, graph_oof, weights)
        sub_pred = blend_frames(base_sub, graph_sub, weights)
        candidates.append(save_candidate(name, train, sub, base_sub, oof_pred, sub_pred, {
            'type': 'base_graph_target_blend',
            'base_tag': base_tag,
            'weights': weights,
        }))

    high_weights = {
        'Q1': 0.22, 'Q2': 0.10, 'Q3': 0.05,
        'S1': 0.25, 'S2': 0.23, 'S3': 0.05, 'S4': 0.14,
    }
    oof_pred, oof_weight_diag = confidence_blend(base_oof, graph_oof, graph_conf_oof, high_weights)
    sub_pred, sub_weight_diag = confidence_blend(base_sub, graph_sub, graph_conf_sub, high_weights)
    candidates.append(save_candidate(
        'v49_base_graph_confidence_guarded',
        train,
        sub,
        base_sub,
        oof_pred,
        sub_pred,
        {
            'type': 'confidence_guarded_graph_blend',
            'base_tag': base_tag,
            'high_weights': high_weights,
            'oof_weight_diag': oof_weight_diag,
            'sub_weight_diag': sub_weight_diag,
        },
    ))

    # A bridge candidate: keep the current public-guided raw policy as base, but
    # let graph move the targets that are weakly trusted by raw (Q3/S3/S4).
    graph_then_raw_weights = {
        'Q1': 0.06, 'Q2': 0.06, 'Q3': 0.14,
        'S1': 0.06, 'S2': 0.06, 'S3': 0.12, 'S4': 0.20,
    }
    oof_pred = blend_frames(base_oof, graph_oof, graph_then_raw_weights)
    sub_pred = blend_frames(base_sub, graph_sub, graph_then_raw_weights)
    candidates.append(save_candidate(
        'v49_base_graph_raw_weak_target_bridge',
        train,
        sub,
        base_sub,
        oof_pred,
        sub_pred,
        {
            'type': 'graph_bridge_for_raw_weak_targets',
            'base_tag': base_tag,
            'weights': graph_then_raw_weights,
        },
    ))

    candidates = sorted(candidates, key=lambda item: item['oof_loss'])
    role_counts = {
        col: int(role_df[col].sum())
        for col in ['is_interior', 'prev_next_agree', 'has_prev', 'has_next']
    }
    summary = {
        'exp_tag': 'v49_subject_sequence_graph_propagation',
        'base': {
            'tag': base_tag,
            'oof_path': base_oof_path,
            'submission_path': base_sub_path,
            'oof_loss': base_loss,
            'oof_per_target': base_per_target,
        },
        'raw': {
            'oof_path': str(RAW_OOF),
            'submission_path': str(RAW_SUB),
            'oof_loss': raw_loss,
            'oof_per_target': raw_per_target,
        },
        'graph': {
            'pseudo_block_lengths': PSEUDO_BLOCK_LENGTHS,
            'oof_loss': graph_loss,
            'oof_per_target': graph_per_target,
            'oof_path': str(graph_oof_path),
            'submission_path': str(graph_sub_path),
            'confidence_oof_path': str(graph_conf_oof_path),
            'confidence_submission_path': str(graph_conf_sub_path),
            'role_path': str(role_path),
            'role_counts': role_counts,
        },
        'candidates': candidates,
        'notes': [
            'OOF here is pseudo-hidden block validation, not random KFold.',
            'If graph candidates hurt public score, keep v48 as score exploit and use v49 only for structural insight.',
            'If graph helps S4 or Q3/S3, v50 should formalize graph/raw target routing.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v49_subject_sequence_graph_propagation.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('[v49] base_oof:', f'{base_loss:.6f}')
    print('[v49] raw_oof:', f'{raw_loss:.6f}')
    print('[v49] graph_pseudo_oof:', f'{graph_loss:.6f}')
    print('[v49] top candidates by pseudo OOF:')
    for item in candidates[:8]:
        print(' ', item['name'], f"oof={item['oof_loss']:.6f}", item['submission'])
    print('[v49] summary:', summary_path)


if __name__ == '__main__':
    main()

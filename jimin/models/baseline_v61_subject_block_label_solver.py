"""v61: subject-block label solver on top of the public-validated v56 anchor.

The failed v57-v59 jumps showed that raw/source ordering is not a trustworthy
public direction.  This experiment instead treats each hidden test segment as a
label-structure puzzle: use visible labels around the segment, target-wise
transition rates, and subject priors to build a calibrated bridge probability.

The solver is evaluated on train by hiding the same subject/run profile as the
test set, then only blended into v56 in controlled role/target policies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v56_block_router as v56


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']
ROLES = ['simple_interior', 'fragmented_interior', 'tail']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_TAG = 'v56_block_router_mid'
EPS = 1e-12


def ensure_dirs() -> None:
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def with_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out['sleep_date'] = pd.to_datetime(out['sleep_date'])
    out['lifelog_date'] = pd.to_datetime(out['lifelog_date'])
    return out


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def label_prob(value: float, prior: float, confidence: float) -> float:
    """Map a hard/soft label mean to a non-extreme probability."""
    calibrated = 0.13 + 0.74 * float(value)
    return float(np.clip(confidence * calibrated + (1.0 - confidence) * prior, 0.02, 0.98))


def estimate_transition_matrices(train: pd.DataFrame, alpha: float = 2.0):
    matrices = {}
    diagnostics = {}
    dated = with_dates(train)
    for target in TARGETS:
        counts = np.full((2, 2), float(alpha))
        n_pairs = 0
        for _, grp in dated.sort_values(['subject_id', 'sleep_date']).groupby('subject_id', sort=False):
            values = grp[target].to_numpy(dtype=int)
            dates = grp['sleep_date'].to_numpy()
            for i in range(len(grp) - 1):
                gap = int((dates[i + 1] - dates[i]) / np.timedelta64(1, 'D'))
                if 1 <= gap <= 2:
                    counts[values[i], values[i + 1]] += 1.0
                    n_pairs += 1
        matrix = counts / counts.sum(axis=1, keepdims=True)
        matrices[target] = matrix
        diagnostics[target] = {
            'n_pairs': int(n_pairs),
            'matrix': np.round(matrix, 6).tolist(),
        }
    return matrices, diagnostics


def matrix_power(matrix: np.ndarray, days: float) -> np.ndarray:
    steps = int(np.clip(round(float(days)), 1, 45))
    return np.linalg.matrix_power(matrix, steps)


def subject_global_means(train: pd.DataFrame, roles: pd.Series | None = None):
    if roles is None:
        known = train
    else:
        known = train.loc[roles == 'visible']
    global_mean = {target: float(known[target].mean()) for target in TARGETS}
    subject_mean = {
        target: known.groupby('subject_id')[target].mean().to_dict()
        for target in TARGETS
    }
    return global_mean, subject_mean


def smoothed_prior(subject_id: str, target: str, global_mean: dict, subject_mean: dict) -> float:
    global_p = float(global_mean[target])
    subject_p = float(subject_mean[target].get(subject_id, global_p))
    return float(np.clip(0.74 * subject_p + 0.26 * global_p, 0.04, 0.96))


def nearest_known(known_positions: np.ndarray, pos: int):
    prev_positions = known_positions[known_positions < pos]
    next_positions = known_positions[known_positions > pos]
    prev_pos = int(prev_positions[-1]) if len(prev_positions) else None
    next_pos = int(next_positions[0]) if len(next_positions) else None
    return prev_pos, next_pos


def recent_mean(seq: pd.DataFrame, known_positions: np.ndarray, target: str, pos: int, side: str):
    if side == 'prev':
        positions = known_positions[known_positions < pos][-3:]
    else:
        positions = known_positions[known_positions > pos][:3]
    if len(positions) == 0:
        return np.nan
    return float(seq.loc[positions, target].mean())


def backward_prob(next_label: int, days: float, prior: float, matrix: np.ndarray) -> float:
    powered = matrix_power(matrix, days)
    num0 = (1.0 - prior) * powered[0, next_label]
    num1 = prior * powered[1, next_label]
    denom = max(num0 + num1, EPS)
    return float(num1 / denom)


def bridge_prob(prev_label: int, next_label: int, d_prev: float, d_next: float, matrix: np.ndarray):
    forward = matrix_power(matrix, d_prev)
    backward = matrix_power(matrix, d_next)
    num0 = forward[prev_label, 0] * backward[0, next_label]
    num1 = forward[prev_label, 1] * backward[1, next_label]
    denom = max(num0 + num1, EPS)
    return float(num1 / denom)


def solver_for_row(
    seq: pd.DataFrame,
    known_positions: np.ndarray,
    pos: int,
    target: str,
    global_mean: dict,
    subject_mean: dict,
    transitions: dict,
):
    row = seq.loc[pos]
    subject_id = row['subject_id']
    prior = smoothed_prior(subject_id, target, global_mean, subject_mean)
    matrix = transitions[target]
    prev_pos, next_pos = nearest_known(known_positions, pos)

    prev_label = np.nan
    next_label = np.nan
    dist_prev = np.nan
    dist_next = np.nan
    is_interior = prev_pos is not None and next_pos is not None

    if prev_pos is not None:
        prev_label = int(seq.loc[prev_pos, target])
        dist_prev = max(1.0, float((row['sleep_date'] - seq.loc[prev_pos, 'sleep_date']).days))
    if next_pos is not None:
        next_label = int(seq.loc[next_pos, target])
        dist_next = max(1.0, float((seq.loc[next_pos, 'sleep_date'] - row['sleep_date']).days))

    prev_recent = recent_mean(seq, known_positions, target, pos, 'prev')
    next_recent = recent_mean(seq, known_positions, target, pos, 'next')
    recent_parts = [x for x in [prev_recent, next_recent] if np.isfinite(x)]
    recent_value = float(np.mean(recent_parts)) if recent_parts else prior
    recent = label_prob(recent_value, prior, 0.58)

    if is_interior:
        span = max(1.0, dist_prev + dist_next)
        frac = float(np.clip(dist_prev / span, 0.0, 1.0))
        linear_value = (1.0 - frac) * prev_label + frac * next_label
        w_prev = 1.0 / max(dist_prev, 1.0)
        w_next = 1.0 / max(dist_next, 1.0)
        invdist_value = float((w_prev * prev_label + w_next * next_label) / (w_prev + w_next))
        markov = bridge_prob(prev_label, next_label, dist_prev, dist_next, matrix)
        linear = label_prob(linear_value, prior, 0.74)
        invdist = label_prob(invdist_value, prior, 0.72)
        agree = bool(prev_label == next_label)
        if agree:
            pred = 0.50 * markov + 0.20 * linear + 0.16 * invdist + 0.09 * recent + 0.05 * prior
            confidence = 0.72 / np.sqrt(1.0 + 0.04 * span)
        else:
            pred = 0.44 * markov + 0.24 * linear + 0.12 * invdist + 0.10 * recent + 0.10 * prior
            confidence = 0.48 / np.sqrt(1.0 + 0.03 * span)
        meta = {
            'has_prev': True,
            'has_next': True,
            'prev_next_agree': agree,
            'dist_prev': dist_prev,
            'dist_next': dist_next,
            'markov': markov,
        }
    elif prev_pos is not None:
        forward = matrix_power(matrix, dist_prev)[prev_label, 1]
        prev_recent_value = prev_recent if np.isfinite(prev_recent) else prev_label
        recent = label_prob(prev_recent_value, prior, 0.62)
        pred = 0.56 * forward + 0.24 * recent + 0.20 * prior
        confidence = max(0.18, min(0.55, 0.63 / np.sqrt(max(1.0, dist_prev))))
        meta = {
            'has_prev': True,
            'has_next': False,
            'prev_next_agree': False,
            'dist_prev': dist_prev,
            'dist_next': np.nan,
            'markov': float(forward),
        }
    elif next_pos is not None:
        backward = backward_prob(next_label, dist_next, prior, matrix)
        next_recent_value = next_recent if np.isfinite(next_recent) else next_label
        recent = label_prob(next_recent_value, prior, 0.62)
        pred = 0.56 * backward + 0.24 * recent + 0.20 * prior
        confidence = max(0.18, min(0.55, 0.63 / np.sqrt(max(1.0, dist_next))))
        meta = {
            'has_prev': False,
            'has_next': True,
            'prev_next_agree': False,
            'dist_prev': np.nan,
            'dist_next': dist_next,
            'markov': float(backward),
        }
    else:
        pred = prior
        confidence = 0.12
        meta = {
            'has_prev': False,
            'has_next': False,
            'prev_next_agree': False,
            'dist_prev': np.nan,
            'dist_next': np.nan,
            'markov': prior,
        }

    return float(np.clip(pred, 0.03, 0.97)), float(np.clip(confidence, 0.0, 1.0)), meta


def train_hidden_runs(train: pd.DataFrame, profiles: dict):
    runs_by_subject = {}
    role = pd.Series('visible', index=train.index, dtype=object)
    for sid, grp in train.groupby('subject_id', sort=True):
        profile = profiles[sid]
        idx = grp.sort_values('sleep_date').index.to_numpy()
        x_lengths = profile['x_runs']
        hidden_total = int(sum(x_lengths))
        visible_total = len(idx) - hidden_total
        gaps = v56.proportional_gaps(visible_total, profile['t_runs'])
        interior_role = 'simple_interior' if profile['is_simple'] else 'fragmented_interior'
        cursor = 0
        subject_runs = []
        for run_i, x_len in enumerate(x_lengths):
            cursor += gaps[run_i]
            selected = idx[cursor:cursor + x_len]
            run_role = 'tail' if run_i == len(x_lengths) - 1 else interior_role
            role.loc[selected] = run_role
            subject_runs.append({
                'run_id': run_i,
                'role': run_role,
                'indices': selected.tolist(),
            })
            cursor += x_len
        runs_by_subject[sid] = subject_runs
    return runs_by_subject, role


def build_pseudo_solver(train: pd.DataFrame, profiles: dict, train_roles: pd.Series):
    dated = with_dates(train)
    transitions, transition_diag = estimate_transition_matrices(train.loc[train_roles == 'visible'])
    global_mean, subject_mean = subject_global_means(train, train_roles)
    runs_by_subject, rebuilt_roles = train_hidden_runs(train, profiles)
    if not rebuilt_roles.equals(train_roles):
        raise ValueError('Rebuilt train roles do not match v56 roles.')

    solver = train[KEYS].copy()
    confidence = train[KEYS].copy()
    meta_rows = []
    for target in TARGETS:
        solver[target] = np.nan
        confidence[target] = 0.0

    for sid, grp in dated.sort_values(['subject_id', 'sleep_date']).groupby('subject_id', sort=False):
        grp = grp.reset_index(drop=False).rename(columns={'index': '_orig_index'})
        hidden_orig = {
            int(idx)
            for run in runs_by_subject[sid]
            for idx in run['indices']
        }
        known_positions = grp.index[~grp['_orig_index'].isin(hidden_orig)].to_numpy(dtype=int)
        run_lookup = {}
        for run in runs_by_subject[sid]:
            for offset, orig_idx in enumerate(run['indices']):
                run_lookup[int(orig_idx)] = {
                    'run_id': run['run_id'],
                    'role': run['role'],
                    'run_len': len(run['indices']),
                    'pos_in_run': offset,
                }

        for pos, row in grp.iterrows():
            orig_idx = int(row['_orig_index'])
            if orig_idx not in hidden_orig:
                for target in TARGETS:
                    solver.loc[orig_idx, target] = np.nan
                continue
            run_info = run_lookup[orig_idx]
            rec = {
                'row_index': orig_idx,
                'subject_id': sid,
                'role': run_info['role'],
                'run_id': int(run_info['run_id']),
                'run_len': int(run_info['run_len']),
                'pos_in_run': int(run_info['pos_in_run']),
            }
            for target in TARGETS:
                pred, conf, meta = solver_for_row(
                    grp,
                    known_positions,
                    int(pos),
                    target,
                    global_mean,
                    subject_mean,
                    transitions,
                )
                solver.loc[orig_idx, target] = pred
                confidence.loc[orig_idx, target] = conf
                if target == 'Q1':
                    rec.update({
                        'has_prev': bool(meta['has_prev']),
                        'has_next': bool(meta['has_next']),
                        'prev_next_agree': bool(meta['prev_next_agree']),
                        'dist_prev': None if not np.isfinite(meta['dist_prev']) else float(meta['dist_prev']),
                        'dist_next': None if not np.isfinite(meta['dist_next']) else float(meta['dist_next']),
                    })
            meta_rows.append(rec)

    return solver, confidence, pd.DataFrame(meta_rows), transition_diag


def build_test_solver(train: pd.DataFrame, sub: pd.DataFrame, profiles: dict):
    train_dated = with_dates(train)
    sub_dated = with_dates(sub)
    transitions, transition_diag = estimate_transition_matrices(train)
    global_mean, subject_mean = subject_global_means(train)

    solver = sub[KEYS].copy()
    confidence = sub[KEYS].copy()
    meta_rows = []
    for target in TARGETS:
        solver[target] = np.nan
        confidence[target] = 0.0

    for sid in sorted(train['subject_id'].unique()):
        tr = train_dated.loc[train_dated['subject_id'] == sid, KEYS + TARGETS].copy()
        tr['_split'] = 'train'
        tr['_orig_index'] = tr.index
        te = sub_dated.loc[sub_dated['subject_id'] == sid, KEYS].copy()
        for target in TARGETS:
            te[target] = np.nan
        te['_split'] = 'test'
        te['_orig_index'] = te.index
        seq = (
            pd.concat([tr, te], ignore_index=True)
            .sort_values(['sleep_date', 'lifelog_date', '_split'])
            .reset_index(drop=True)
        )
        known_positions = seq.index[seq['_split'] == 'train'].to_numpy(dtype=int)
        profile = profiles[sid]
        interior_role = 'simple_interior' if profile['is_simple'] else 'fragmented_interior'

        run_start = None
        run_id = -1
        for pos, split in enumerate(seq['_split'].tolist() + ['sentinel']):
            if split == 'test' and run_start is None:
                run_start = pos
                run_id += 1
            if split != 'test' and run_start is not None:
                run_end = pos
                run_positions = list(range(run_start, run_end))
                run_role = 'tail' if run_id == profile['n_x_runs'] - 1 else interior_role
                for offset, row_pos in enumerate(run_positions):
                    sub_idx = int(seq.loc[row_pos, '_orig_index'])
                    rec = {
                        'row_index': sub_idx,
                        'subject_id': sid,
                        'role': run_role,
                        'run_id': int(run_id),
                        'run_len': int(len(run_positions)),
                        'pos_in_run': int(offset),
                    }
                    for target in TARGETS:
                        pred, conf, meta = solver_for_row(
                            seq,
                            known_positions,
                            int(row_pos),
                            target,
                            global_mean,
                            subject_mean,
                            transitions,
                        )
                        solver.loc[sub_idx, target] = pred
                        confidence.loc[sub_idx, target] = conf
                        if target == 'Q1':
                            rec.update({
                                'has_prev': bool(meta['has_prev']),
                                'has_next': bool(meta['has_next']),
                                'prev_next_agree': bool(meta['prev_next_agree']),
                                'dist_prev': None if not np.isfinite(meta['dist_prev']) else float(meta['dist_prev']),
                                'dist_next': None if not np.isfinite(meta['dist_next']) else float(meta['dist_next']),
                            })
                    meta_rows.append(rec)
                run_start = None

    solver[TARGETS] = solver[TARGETS].astype(float).apply(clip_prob)
    confidence[TARGETS] = confidence[TARGETS].astype(float).clip(0.0, 1.0)
    return solver, confidence, pd.DataFrame(meta_rows), transition_diag


def fill_solver_visible(anchor: pd.DataFrame, solver: pd.DataFrame):
    out = anchor[KEYS].copy()
    for target in TARGETS:
        out[target] = solver[target].where(solver[target].notna(), anchor[target])
    return out


def role_target_grid(train, anchor_oof, solver_oof, train_roles):
    rows = []
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0]
    for role in ROLES:
        mask = train_roles == role
        for target in TARGETS:
            y = train.loc[mask, target]
            base = anchor_oof.loc[mask, target].to_numpy(dtype=float)
            other = solver_oof.loc[mask, target].to_numpy(dtype=float)
            base_loss = v56.target_logloss(y, base)
            solver_loss = v56.target_logloss(y, other)
            losses = []
            for weight in grid:
                blended = clip_prob((1.0 - weight) * base + weight * other)
                losses.append(v56.target_logloss(y, blended))
            best_i = int(np.argmin(losses))
            rows.append({
                'role': role,
                'target': target,
                'n_rows': int(mask.sum()),
                'anchor_loss': float(base_loss),
                'solver_loss': float(solver_loss),
                'best_weight': float(grid[best_i]),
                'best_loss': float(losses[best_i]),
                'best_delta': float(losses[best_i] - base_loss),
                'solver_delta': float(solver_loss - base_loss),
            })
    return pd.DataFrame(rows)


def negative_role_target_grid(train, anchor_oof, solver_oof, train_roles):
    rows = []
    grid = [-1.25, -1.00, -0.85, -0.70, -0.55, -0.40, -0.30, -0.20, -0.10, -0.05, 0.0]
    for role in ROLES:
        mask = train_roles == role
        for target in TARGETS:
            y = train.loc[mask, target]
            base = anchor_oof.loc[mask, target].to_numpy(dtype=float)
            other = solver_oof.loc[mask, target].to_numpy(dtype=float)
            base_loss = v56.target_logloss(y, base)
            losses = []
            for weight in grid:
                corrected = clip_prob(base + weight * (other - base))
                losses.append(v56.target_logloss(y, corrected))
            best_i = int(np.argmin(losses))
            rows.append({
                'role': role,
                'target': target,
                'n_rows': int(mask.sum()),
                'anchor_loss': float(base_loss),
                'best_weight': float(grid[best_i]),
                'best_loss': float(losses[best_i]),
                'best_delta': float(losses[best_i] - base_loss),
            })
    return pd.DataFrame(rows)


def selected_weights(grid_df: pd.DataFrame, strength: str):
    if strength == 'safe':
        shrink = 0.35
        cap = 0.12
        min_gain = 0.0009
        allow_q3s3 = False
    elif strength == 'mid':
        shrink = 0.58
        cap = 0.24
        min_gain = 0.00045
        allow_q3s3 = False
    elif strength == 'bold':
        shrink = 0.86
        cap = 0.42
        min_gain = 0.00015
        allow_q3s3 = True
    else:
        raise ValueError(strength)

    weights = {role: {target: 0.0 for target in TARGETS} for role in ROLES}
    for row in grid_df.itertuples(index=False):
        if row.target in ['Q3', 'S3'] and not allow_q3s3:
            continue
        if row.target in ['Q3', 'S3'] and row.best_delta > -0.003:
            continue
        if row.best_weight <= 0 or row.best_delta > -min_gain:
            continue
        weights[row.role][row.target] = float(min(cap, row.best_weight * shrink))
    return weights


def selected_anti_weights(neg_grid_df: pd.DataFrame, strength: str):
    if strength == 'safe':
        shrink = 0.18
        cap_abs = 0.18
        min_gain = 0.004
    elif strength == 'mid':
        shrink = 0.32
        cap_abs = 0.34
        min_gain = 0.002
    elif strength == 'bold':
        shrink = 0.52
        cap_abs = 0.58
        min_gain = 0.001
    else:
        raise ValueError(strength)

    weights = {role: {target: 0.0 for target in TARGETS} for role in ROLES}
    for row in neg_grid_df.itertuples(index=False):
        if row.target in ['Q3', 'S3']:
            continue
        if row.best_weight >= 0 or row.best_delta > -min_gain:
            continue
        weights[row.role][row.target] = float(max(-cap_abs, row.best_weight * shrink))
    return weights


PUBLIC_AXIS_WEIGHTS = {
    'simple_interior': {'Q2': 0.00, 'S4': 0.06},
    'fragmented_interior': {'Q2': 0.16, 'S4': 0.12},
    'tail': {'Q2': 0.00, 'S4': 0.12},
}

BOLD_STATIC_WEIGHTS = {
    'simple_interior': {'Q1': 0.12, 'Q2': 0.08, 'S1': 0.12, 'S2': 0.10, 'S4': 0.14},
    'fragmented_interior': {'Q1': 0.10, 'Q2': 0.24, 'S1': 0.10, 'S2': 0.10, 'S4': 0.20},
    'tail': {'Q1': 0.08, 'Q2': 0.06, 'S1': 0.08, 'S2': 0.08, 'S4': 0.18},
}


ANTI_PUBLIC_AXIS_WEIGHTS = {
    'simple_interior': {'Q2': -0.18, 'S4': -0.10},
    'fragmented_interior': {'S4': -0.10},
    'tail': {'Q2': -0.10, 'S4': -0.22},
}


ANTI_CORE_MID_WEIGHTS = {
    'simple_interior': {'Q2': -0.24},
    'fragmented_interior': {'S1': -0.34, 'S2': -0.34},
    'tail': {'Q1': -0.25, 'S4': -0.25},
}

ANTI_CORE_BOLD_WEIGHTS = {
    'simple_interior': {'Q2': -0.40},
    'fragmented_interior': {'S1': -0.55, 'S2': -0.55},
    'tail': {'Q1': -0.38, 'S2': -0.18, 'S4': -0.38},
}


def normalize_weights(weights: dict):
    out = {role: {target: 0.0 for target in TARGETS} for role in ROLES}
    for role, spec in weights.items():
        for target, value in spec.items():
            out[role][target] = float(value)
    return out


def apply_label_bridge(anchor, solver, confidence, roles, weights, cap, confidence_scale=True):
    out = anchor.copy()
    weights = normalize_weights(weights)
    for role in ROLES:
        mask = roles == role
        if not bool(mask.any()):
            continue
        for target in TARGETS:
            base_weight = weights[role].get(target, 0.0)
            if abs(base_weight) <= 0:
                continue
            delta = solver.loc[mask, target].to_numpy(dtype=float) - anchor.loc[mask, target].to_numpy(dtype=float)
            delta = np.clip(delta, -float(cap), float(cap))
            if confidence_scale:
                conf = confidence.loc[mask, target].to_numpy(dtype=float)
                row_weight = base_weight * (0.40 + 0.60 * conf)
            else:
                row_weight = np.full(np.sum(mask), base_weight, dtype=float)
            out.loc[mask, target] = clip_prob(anchor.loc[mask, target].to_numpy(dtype=float) + row_weight * delta)
    return out


def describe_solver(train, anchor_oof, solver_oof, roles):
    diagnostics = {
        'hidden_solver': v56.evaluate(train, solver_oof, roles != 'visible'),
        'anchor_hidden': v56.evaluate(train, anchor_oof, roles != 'visible'),
        'role_solver': {},
        'role_anchor': {},
    }
    for role in ROLES:
        diagnostics['role_solver'][role] = v56.evaluate(train, solver_oof, roles == role)
        diagnostics['role_anchor'][role] = v56.evaluate(train, anchor_oof, roles == role)
    return diagnostics


def save_frame(path: Path, frame: pd.DataFrame):
    out = frame.copy()
    out[TARGETS] = out[TARGETS].astype(float).apply(clip_prob)
    out.to_csv(path, index=False)


def save_candidate(
    name: str,
    note: str,
    train: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    solver_oof: pd.DataFrame,
    solver_sub: pd.DataFrame,
    conf_oof: pd.DataFrame,
    conf_sub: pd.DataFrame,
    train_roles: pd.Series,
    test_roles: pd.Series,
    weights: dict,
    cap: float,
    confidence_scale: bool = True,
):
    oof = apply_label_bridge(anchor_oof, solver_oof, conf_oof, train_roles, weights, cap, confidence_scale)
    submission = apply_label_bridge(anchor_sub, solver_sub, conf_sub, test_roles, weights, cap, confidence_scale)
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    save_frame(oof_path, oof)
    save_frame(sub_path, submission)
    return {
        'name': name,
        'note': note,
        'weights': normalize_weights(weights),
        'cap': float(cap),
        'confidence_scale': bool(confidence_scale),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
        'full_oof': v56.evaluate(train, oof),
        'role_oof': v56.role_evaluations(train, oof, train_roles),
        'distribution_vs_anchor': v56.describe_vs_anchor(submission, anchor_sub, test_roles),
    }


def main():
    ensure_dirs()
    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_SAMPLE_PATH)

    profiles, test_roles = v56.build_test_profiles(train, sub)
    train_roles = v56.build_train_roles(train, profiles)
    anchor_oof = v56.load_oof(ANCHOR_TAG, train)
    anchor_sub = v56.load_submission(ANCHOR_TAG, sub)

    pseudo_solver, pseudo_conf, pseudo_meta, pseudo_transition_diag = build_pseudo_solver(
        train,
        profiles,
        train_roles,
    )
    test_solver, test_conf, test_meta, test_transition_diag = build_test_solver(train, sub, profiles)

    solver_oof = fill_solver_visible(anchor_oof, pseudo_solver)
    solver_sub = fill_solver_visible(anchor_sub, test_solver)
    save_frame(OOF_DIR / 'oof_v61_label_solver_raw.csv', solver_oof)
    save_frame(SUB_DIR / 'submission_v61_label_solver_raw.csv', solver_sub)
    save_frame(OOF_DIR / 'oof_v61_label_solver_confidence.csv', pseudo_conf)
    save_frame(SUB_DIR / 'submission_v61_label_solver_confidence.csv', test_conf)
    pseudo_meta.to_csv(SUMMARY_DIR / 'v61_pseudo_hidden_meta.csv', index=False)
    test_meta.to_csv(SUMMARY_DIR / 'v61_test_hidden_meta.csv', index=False)

    grid_df = role_target_grid(train, anchor_oof, solver_oof, train_roles)
    grid_path = SUMMARY_DIR / 'v61_role_target_grid.csv'
    grid_df.to_csv(grid_path, index=False)
    neg_grid_df = negative_role_target_grid(train, anchor_oof, solver_oof, train_roles)
    neg_grid_path = SUMMARY_DIR / 'v61_negative_role_target_grid.csv'
    neg_grid_df.to_csv(neg_grid_path, index=False)

    safe_weights = selected_weights(grid_df, 'safe')
    mid_weights = selected_weights(grid_df, 'mid')
    bold_weights = selected_weights(grid_df, 'bold')
    anti_safe_weights = selected_anti_weights(neg_grid_df, 'safe')
    anti_mid_weights = selected_anti_weights(neg_grid_df, 'mid')
    anti_bold_weights = selected_anti_weights(neg_grid_df, 'bold')

    candidates = [
        save_candidate(
            'v61_label_bridge_public_axis_safe',
            'Tiny label bridge only on the public-validated Q2/S4 axis.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            PUBLIC_AXIS_WEIGHTS,
            cap=0.055,
        ),
        save_candidate(
            'v61_label_bridge_oof_selected_safe',
            'OOF-selected non-Q3/S3 role-target bridge with heavy shrinkage.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            safe_weights,
            cap=0.055,
        ),
        save_candidate(
            'v61_label_bridge_oof_selected_mid',
            'OOF-selected non-Q3/S3 role-target bridge at moderate shrinkage.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            mid_weights,
            cap=0.085,
        ),
        save_candidate(
            'v61_label_bridge_oof_selected_bold',
            'OOF-selected bold bridge; Q3/S3 allowed only if their role-target gain is large.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            bold_weights,
            cap=0.125,
        ),
        save_candidate(
            'v61_label_bridge_static_bold_no_q3s3',
            'Manual bigger jump on Q1/Q2/S1/S2/S4, keeping Q3/S3 frozen.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            BOLD_STATIC_WEIGHTS,
            cap=0.115,
        ),
        save_candidate(
            'v61_anti_label_public_axis_mid',
            'Move away from the label solver only on Q2/S4 public-axis roles.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            ANTI_PUBLIC_AXIS_WEIGHTS,
            cap=0.090,
        ),
        save_candidate(
            'v61_anti_label_core_mid',
            'Focused anti-label jump on the strongest pseudo roles: simple Q2, fragmented S1/S2, tail Q1/S4.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            ANTI_CORE_MID_WEIGHTS,
            cap=0.160,
        ),
        save_candidate(
            'v61_anti_label_core_bold_no_q3s3',
            'Bolder focused anti-label jump; Q3/S3 remain frozen.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            ANTI_CORE_BOLD_WEIGHTS,
            cap=0.160,
        ),
        save_candidate(
            'v61_anti_label_oof_selected_safe',
            'OOF-selected anti-label correction with strong shrinkage, Q3/S3 frozen.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            anti_safe_weights,
            cap=0.080,
        ),
        save_candidate(
            'v61_anti_label_oof_selected_mid',
            'OOF-selected anti-label correction at moderate shrinkage, Q3/S3 frozen.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            anti_mid_weights,
            cap=0.115,
        ),
        save_candidate(
            'v61_anti_label_oof_selected_bold_no_q3s3',
            'Big-jump anti-label correction on OOF-positive roles, keeping Q3/S3 frozen.',
            train,
            anchor_oof,
            anchor_sub,
            solver_oof,
            solver_sub,
            pseudo_conf,
            test_conf,
            train_roles,
            test_roles,
            anti_bold_weights,
            cap=0.160,
        ),
    ]

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    solver_diag = describe_solver(train, anchor_oof, solver_oof, train_roles)

    sorted_candidates = sorted(candidates, key=lambda item: item['full_oof']['loss'])
    summary = {
        'exp_tag': 'v61_subject_block_label_solver',
        'anchor': {
            'tag': ANCHOR_TAG,
            'known_public_score': 0.5798876532,
            'eval': anchor_eval,
        },
        'solver': {
            'raw_oof_path': str(OOF_DIR / 'oof_v61_label_solver_raw.csv'),
            'raw_submission_path': str(SUB_DIR / 'submission_v61_label_solver_raw.csv'),
            'confidence_oof_path': str(OOF_DIR / 'oof_v61_label_solver_confidence.csv'),
            'confidence_submission_path': str(SUB_DIR / 'submission_v61_label_solver_confidence.csv'),
            'diagnostics': solver_diag,
            'pseudo_transition_matrices': pseudo_transition_diag,
            'test_transition_matrices': test_transition_diag,
        },
        'role_counts': {
            'train_pseudo': train_roles.value_counts().astype(int).to_dict(),
            'test': test_roles.value_counts().astype(int).to_dict(),
        },
        'grid_path': str(grid_path),
        'negative_grid_path': str(neg_grid_path),
        'role_target_grid': grid_df.to_dict(orient='records'),
        'negative_role_target_grid': neg_grid_df.to_dict(orient='records'),
        'learned_weights': {
            'safe': safe_weights,
            'mid': mid_weights,
            'bold': bold_weights,
            'anti_safe': anti_safe_weights,
            'anti_mid': anti_mid_weights,
            'anti_bold': anti_bold_weights,
        },
        'candidates': candidates,
        'recommended_submit_order': [item['name'] for item in sorted_candidates[:3]],
        'policy_notes': [
            'Raw label solver is saved for inspection but is not recommended as a direct submission.',
            'Public-axis candidate only touches the Q2/S4 families that v56 validated publicly.',
            'OOF-selected safe/mid freeze Q3/S3 because no_q3s3 remains the only public-confirmed direction.',
            'Bold allows Q3/S3 only when the role-target pseudo gain is unusually large.',
            'Anti-label candidates are intentionally risky: they exploit that the label solver is systematically wrong on pseudo-hidden blocks.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v61_subject_block_label_solver.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v61] summary={summary_path}')
    print(f'[v61] grid={grid_path}')
    print(f'[v61] negative_grid={neg_grid_path}')
    print(f'[v61] anchor_full={anchor_eval["full_oof"]["loss"]:.6f} '
          f'anchor_routed={anchor_eval["role_oof"]["routed_rows"]["loss"]:.6f}')
    print(f'[v61] solver_hidden={solver_diag["hidden_solver"]["loss"]:.6f} '
          f'anchor_hidden={solver_diag["anchor_hidden"]["loss"]:.6f}')
    anchor_routed = anchor_eval['role_oof']['routed_rows']['loss']
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        full = item['full_oof']['loss']
        mad = item['distribution_vs_anchor']['mad_vs_anchor']
        print(
            f"  {item['name']}: full_oof={full:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed={routed - anchor_routed:+.6f} "
            f"sub_mad={mad:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

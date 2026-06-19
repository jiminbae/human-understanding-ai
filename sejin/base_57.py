# v57: per-target independent Optuna studies (3-way ensemble).
#
# v56 findings:
#   - 3-way (BASE+MID+BEST) beats 2-way clearly (0.5377 vs 0.5432)
#   - Q1 biggest gainer (-0.032), Q1/S2 benefit most from BEST
#   - Q2 helpful at high weights (alpha~1.17, beta~1.15)
#   - Q3/S3 locked to 0.0 (confirmed harmful)
#   - S4: MID stronger than BEST (alpha>beta)
#
# Key upgrade over v56:
#   Each target gets its OWN Optuna study with its own loss function.
#   → No cross-target noise during search
#   → Each target finds its true optimum independently
#   → Per-target n_trials budget is configurable
#
# Output: one best (alpha, beta) per target → assembled into final blend.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── constants ──────────────────────────────────────────────────────────────────
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS    = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR    = Path(__file__).resolve().parent
TRAIN_PATH  = BASE_DIR / 'ch2026_metrics_train.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR     = OUTPUTS_DIR / 'submissions'
OOF_DIR     = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

BASE_TAG = 'v48_target_delta_scaled_avg430_q2cap115_q3s3guard'
MID_TAG  = 'v53_public_mid_probe'
BEST_TAG = 'v56_optuna_3way'

PUBLIC_SCORES = {
    BASE_TAG:                    0.5805824813,
    MID_TAG:                     0.5800163708,
    'v54_public_mid_s4_half':    0.5800565394,
    'v55_no_q3s3_scale108':      0.5799096236,
}

# v56 3-way reference weights (for warm-start and comparison)
V56_3WAY_WEIGHTS = {
    'Q1': {'alpha': 0.9951, 'beta': 1.1079},
    'Q2': {'alpha': 1.1656, 'beta': 1.1549},
    'Q3': {'alpha': 0.0,    'beta': 0.0},
    'S1': {'alpha': 0.7667, 'beta': 0.9766},
    'S2': {'alpha': 0.9966, 'beta': 1.1903},
    'S3': {'alpha': 0.0,    'beta': 0.0},
    'S4': {'alpha': 1.0702, 'beta': 0.9606},
}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_pair(tag: str):
    return (
        pd.read_csv(OOF_DIR / f'oof_{tag}.csv'),
        pd.read_csv(SUB_DIR / f'submission_{tag}.csv'),
    )


def target_logloss(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def clip_col(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.02, 0.98)


def blend_col(base_col, mid_col, best_col, alpha: float, beta: float) -> np.ndarray:
    return clip_col(base_col + alpha * (mid_col - base_col) + beta * (best_col - base_col))


def evaluate_mean(train: pd.DataFrame, pred: pd.DataFrame) -> float:
    losses = [target_logloss(train[t], pred[t]) for t in TARGETS]
    return float(np.mean(losses))


def describe(pred: pd.DataFrame, base: pd.DataFrame) -> dict:
    diff = pred[TARGETS] - base[TARGETS]
    return {
        'mad_vs_base':     float(diff.abs().to_numpy().mean()),
        'max_abs_vs_base': float(diff.abs().to_numpy().max()),
        'per_target_mad':  {t: float(diff[t].abs().mean()) for t in TARGETS},
        'mean_delta':      {t: float(diff[t].mean()) for t in TARGETS},
        'means':           {t: float(pred[t].mean()) for t in TARGETS},
    }


# ── per-target optimisation ────────────────────────────────────────────────────

def optimise_target(
    target: str,
    y_true: np.ndarray,
    base_col: np.ndarray,
    mid_col: np.ndarray,
    best_col: np.ndarray,
    n_trials: int,
    w_min: float,
    w_max: float,
    seed: int,
    warmstart: dict | None = None,
) -> tuple[float, float, float]:
    """
    Run an independent Optuna study for a single target.
    Returns (best_alpha, best_beta, best_loss).
    """
    # locked targets: skip search entirely
    if target in ('Q3', 'S3'):
        loss = target_logloss(y_true, clip_col(base_col))
        return 0.0, 0.0, loss

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float('alpha', w_min, w_max)
        beta  = trial.suggest_float('beta',  w_min, w_max)
        pred  = blend_col(base_col, mid_col, best_col, alpha, beta)
        return target_logloss(y_true, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction='minimize', sampler=sampler)

    # warm-start: add v56 best as an initial guess
    if warmstart:
        study.enqueue_trial({'alpha': warmstart['alpha'], 'beta': warmstart['beta']})

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    return best.params['alpha'], best.params['beta'], best.value


# ── assemble final blend ───────────────────────────────────────────────────────

def assemble(
    base: pd.DataFrame,
    mid: pd.DataFrame,
    best: pd.DataFrame,
    alphas: dict,
    betas: dict,
) -> pd.DataFrame:
    out = base[KEYS].copy()
    for t in TARGETS:
        out[t] = blend_col(
            base[t].to_numpy(),
            mid[t].to_numpy(),
            best[t].to_numpy(),
            alphas[t], betas[t],
        )
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='v57 per-target independent Optuna')
    parser.add_argument('--n_trials', type=int,   default=500,
                        help='Optuna trials PER TARGET (default 500)')
    parser.add_argument('--w_min',    type=float, default=0.0,
                        help='Lower bound for blend weights (default 0.0)')
    parser.add_argument('--w_max',    type=float, default=1.5,
                        help='Upper bound for blend weights (default 1.5)')
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--no_warmstart', action='store_true',
                        help='Disable v56 warm-start enqueue')
    args = parser.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    train    = pd.read_csv(TRAIN_PATH)
    base_oof, base_sub = load_pair(BASE_TAG)
    mid_oof,  mid_sub  = load_pair(MID_TAG)
    best_oof, best_sub = load_pair(BEST_TAG)

    # ── per-target search ─────────────────────────────────────────────────────
    print(f'[v57] per-target study  n_trials={args.n_trials}  '
          f'w=[{args.w_min}, {args.w_max}]  seed={args.seed}')

    alphas, betas, per_target_loss, per_target_best = {}, {}, {}, {}

    for t in TARGETS:
        warmstart = (None if args.no_warmstart
                     else V56_3WAY_WEIGHTS.get(t))
        a, b, loss = optimise_target(
            target   = t,
            y_true   = train[t].to_numpy(),
            base_col = base_oof[t].to_numpy(),
            mid_col  = mid_oof[t].to_numpy(),
            best_col = best_oof[t].to_numpy(),
            n_trials = args.n_trials,
            w_min    = args.w_min,
            w_max    = args.w_max,
            seed     = args.seed,
            warmstart= warmstart,
        )
        alphas[t] = a
        betas[t]  = b
        per_target_loss[t] = loss
        per_target_best[t] = {'alpha': a, 'beta': b, 'loss': loss}
        tag = '(locked)' if t in ('Q3', 'S3') else f'α={a:.4f} β={b:.4f}'
        print(f'  {t}: loss={loss:.6f}  {tag}')

    # ── assemble and save ─────────────────────────────────────────────────────
    oof_blend = assemble(base_oof, mid_oof, best_oof, alphas, betas)
    sub_blend = assemble(base_sub, mid_sub, best_sub, alphas, betas)

    name     = 'v57_per_target_3way'
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof_blend.to_csv(oof_path, index=False)
    sub_blend.to_csv(sub_path, index=False)

    mean_loss  = evaluate_mean(train, oof_blend)
    dist_info  = describe(sub_blend, base_sub)

    # ── per-target delta table vs base & vs v56 ───────────────────────────────
    base_per = {t: target_logloss(train[t], base_oof[t]) for t in TARGETS}
    v56_per  = {   # from summary JSON
        'Q1': 0.5152993659883601, 'Q2': 0.5555987798167102,
        'Q3': 0.5858722183973512, 'S1': 0.4777881782352393,
        'S2': 0.5096795553824248, 'S3': 0.5314078513947333,
        'S4': 0.5880498839108822,
    }
    delta_vs_base = {t: round(per_target_loss[t] - base_per[t], 6) for t in TARGETS}
    delta_vs_v56  = {t: round(per_target_loss[t] - v56_per[t],  6) for t in TARGETS}

    print(f'\n[v57] assembled  proxy_loss={mean_loss:.6f}')
    print(f'  v56_3way was            0.537671')
    delta_global = mean_loss - 0.5376708333036716
    sign = '+' if delta_global >= 0 else ''
    print(f'  Δ vs v56_3way = {sign}{delta_global:.6f}')
    print(f'\n  per-target Δ vs v56_3way:')
    for t in TARGETS:
        s = '+' if delta_vs_v56[t] >= 0 else ''
        print(f'    {t}: {s}{delta_vs_v56[t]:.6f}')

    # ── also run a joint-study version as comparison ──────────────────────────
    # (re-uses v56 3way logic but with expanded w_max & more trials for fairness)
    print(f'\n[v57] joint study (expanded budget, for comparison) …')
    import optuna as _optuna

    def joint_objective(trial: _optuna.Trial) -> float:
        a_map, b_map = {}, {}
        for t in TARGETS:
            if t in ('Q3', 'S3'):
                a_map[t] = b_map[t] = 0.0
            else:
                a_map[t] = trial.suggest_float(f'alpha_{t}', args.w_min, args.w_max)
                b_map[t] = trial.suggest_float(f'beta_{t}',  args.w_min, args.w_max)
        pred = assemble(base_oof, mid_oof, best_oof, a_map, b_map)
        return evaluate_mean(train, pred)

    joint_study = _optuna.create_study(
        direction='minimize',
        sampler=_optuna.samplers.TPESampler(seed=args.seed),
    )
    # warm-start joint with v57 per-target solution
    joint_study.enqueue_trial(
        {f'alpha_{t}': alphas[t] for t in TARGETS if t not in ('Q3','S3')} |
        {f'beta_{t}':  betas[t]  for t in TARGETS if t not in ('Q3','S3')}
    )
    joint_study.optimize(joint_objective,
                         n_trials=args.n_trials,
                         show_progress_bar=False)

    j_best  = joint_study.best_trial
    j_alpha = {t: j_best.params.get(f'alpha_{t}', 0.0) for t in TARGETS}
    j_beta  = {t: j_best.params.get(f'beta_{t}',  0.0) for t in TARGETS}

    oof_j = assemble(base_oof, mid_oof, best_oof, j_alpha, j_beta)
    sub_j = assemble(base_sub, mid_sub, best_sub, j_alpha, j_beta)
    name_j  = 'v57_joint_expanded'
    oof_j.to_csv(OOF_DIR / f'oof_{name_j}.csv',   index=False)
    sub_j.to_csv(SUB_DIR / f'submission_{name_j}.csv', index=False)
    loss_j  = evaluate_mean(train, oof_j)
    print(f'  joint expanded proxy_loss={loss_j:.6f}')

    # ── summary ───────────────────────────────────────────────────────────────
    candidates = sorted([
        {
            'name':                   name,
            'mode':                   'per_target_independent',
            'oof_proxy_loss':         mean_loss,
            'oof_proxy_per_target':   per_target_loss,
            'per_target_weights':     per_target_best,
            'delta_vs_base':          delta_vs_base,
            'delta_vs_v56_3way':      delta_vs_v56,
            'distribution_vs_base':   dist_info,
            'oof_path':               str(oof_path),
            'submission':             str(sub_path),
        },
        {
            'name':                   name_j,
            'mode':                   'joint_expanded',
            'oof_proxy_loss':         loss_j,
            'weights_alpha':          j_alpha,
            'weights_beta':           j_beta,
            'distribution_vs_base':   describe(sub_j, base_sub),
            'oof_path':               str(OOF_DIR / f'oof_{name_j}.csv'),
            'submission':             str(SUB_DIR / f'submission_{name_j}.csv'),
        },
    ], key=lambda x: x['oof_proxy_loss'])

    summary = {
        'exp_tag':              'v57_per_target_study',
        'optuna_config':        vars(args),
        'known_public_scores':  PUBLIC_SCORES,
        'v56_3way_oof_loss':    0.5376708333036716,
        'base_oof_per_target':  base_per,
        'recommended_submit_order': [c['name'] for c in candidates],
        'candidates':           candidates,
    }
    path = SUMMARY_DIR / 'summary_v57_per_target_study.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'\n[v57] summary → {path}')
    print('  Final ranking:')
    for c in candidates:
        d = c['oof_proxy_loss'] - PUBLIC_SCORES[BASE_TAG]
        s = '+' if d >= 0 else ''
        print(f"    {c['name']}: proxy={c['oof_proxy_loss']:.6f}  Δbase={s}{d:.6f}")
    print(f'\n  → Submit first: {candidates[0]["name"]}')
    print(f"    {candidates[0]['submission']}")


if __name__ == '__main__':
    main()
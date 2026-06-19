# v56: Optuna per-target weight optimisation + 3-way ensemble (BASE + MID + BEST).
#
# Builds on v55 findings:
#   - Q3 / S3 deltas are harmful → weight capped at 0.0 in search unless --allow_q3s3
#   - S4 at 1.0 (full delta) is validated; s4_half worsened
#   - Q2 status still unclear → searched freely [0, 1.2]
#
# Two modes are run back-to-back:
#   A) 2-way optimisation  : alpha_t in [0, 1.2] for each target   (BASE→MID blend)
#   B) 3-way optimisation  : alpha_t (BASE→MID), beta_t (BASE→BEST)
#      final = BASE + alpha*(MID-BASE) + beta*(BEST-BASE)
#      This lets BEST contribute independently on targets where it is best.
#
# Optuna minimises OOF proxy log-loss with TPE sampler.
# Best candidates are saved as submission + OOF CSVs.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ──────────────────────────────────────────────────────────────────────
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS    = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR   = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR     = OUTPUTS_DIR / 'submissions'
OOF_DIR     = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

BASE_TAG = 'v48_target_delta_scaled_avg430_q2cap115_q3s3guard'
MID_TAG  = 'v53_public_mid_probe'
BEST_TAG = 'v55_no_q3s3_scale108'

# Known public leaderboard scores (lower = better)
PUBLIC_SCORES = {
    BASE_TAG:                    0.5805824813,
    MID_TAG:                     0.5800163708,
    'v54_public_mid_s4_half':    0.5800565394,
    BEST_TAG:                    0.5799096236,
}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_pair(tag: str):
    oof = pd.read_csv(OOF_DIR / f'oof_{tag}.csv')
    sub = pd.read_csv(SUB_DIR / f'submission_{tag}.csv')
    return oof, sub


def clip_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[TARGETS] = out[TARGETS].clip(0.02, 0.98)
    return out


def target_logloss(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate(train: pd.DataFrame, pred: pd.DataFrame):
    per = {t: target_logloss(train[t], pred[t]) for t in TARGETS}
    return float(np.mean(list(per.values()))), per


def describe(pred: pd.DataFrame, base: pd.DataFrame) -> dict:
    diff = pred[TARGETS] - base[TARGETS]
    return {
        'mad_vs_base':      float(diff.abs().to_numpy().mean()),
        'max_abs_vs_base':  float(diff.abs().to_numpy().max()),
        'per_target_mad':   {t: float(diff[t].abs().mean()) for t in TARGETS},
        'mean_delta':       {t: float(diff[t].mean()) for t in TARGETS},
        'means':            {t: float(pred[t].mean()) for t in TARGETS},
    }


# ── 2-way blend  BASE + alpha*(MID-BASE) ──────────────────────────────────────

def blend_2way(base: pd.DataFrame, mid: pd.DataFrame, alphas: dict) -> pd.DataFrame:
    out = base[KEYS].copy()
    for t in TARGETS:
        out[t] = base[t] + float(alphas[t]) * (mid[t] - base[t])
    return clip_frame(out)


def make_objective_2way(train, base_oof, mid_oof, allow_q3s3: bool, w_max: float):
    """Return an Optuna objective for 2-way blending."""
    def objective(trial: optuna.Trial) -> float:
        alphas = {}
        for t in TARGETS:
            if t in ('Q3', 'S3') and not allow_q3s3:
                alphas[t] = 0.0
            else:
                alphas[t] = trial.suggest_float(f'alpha_{t}', 0.0, w_max)
        pred = blend_2way(base_oof, mid_oof, alphas)
        loss, _ = evaluate(train, pred)
        return loss
    return objective


# ── 3-way blend  BASE + alpha*(MID-BASE) + beta*(BEST-BASE) ───────────────────

def blend_3way(base: pd.DataFrame, mid: pd.DataFrame, best: pd.DataFrame,
               alphas: dict, betas: dict) -> pd.DataFrame:
    out = base[KEYS].copy()
    for t in TARGETS:
        out[t] = (base[t]
                  + float(alphas[t]) * (mid[t]  - base[t])
                  + float(betas[t])  * (best[t] - base[t]))
    return clip_frame(out)


def make_objective_3way(train, base_oof, mid_oof, best_oof,
                        allow_q3s3: bool, w_max: float):
    """Return an Optuna objective for 3-way blending."""
    def objective(trial: optuna.Trial) -> float:
        alphas, betas = {}, {}
        for t in TARGETS:
            if t in ('Q3', 'S3') and not allow_q3s3:
                alphas[t] = 0.0
                betas[t]  = 0.0
            else:
                alphas[t] = trial.suggest_float(f'alpha_{t}', 0.0, w_max)
                betas[t]  = trial.suggest_float(f'beta_{t}',  0.0, w_max)
        pred = blend_3way(base_oof, mid_oof, best_oof, alphas, betas)
        loss, _ = evaluate(train, pred)
        return loss
    return objective


# ── save helpers ───────────────────────────────────────────────────────────────

def save_result(name: str, train: pd.DataFrame,
                oof: pd.DataFrame, sub: pd.DataFrame,
                base_oof: pd.DataFrame, base_sub: pd.DataFrame,
                weights: dict, note: str) -> dict:
    oof_path = OOF_DIR  / f'oof_{name}.csv'
    sub_path = SUB_DIR  / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    sub.to_csv(sub_path, index=False)
    loss, per = evaluate(train, oof)
    return {
        'name':                name,
        'weights':             weights,
        'note':                note,
        'oof_proxy_loss':      loss,
        'oof_proxy_per_target': per,
        'distribution_vs_base': describe(sub, base_sub),
        'oof_path':            str(oof_path),
        'submission':          str(sub_path),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='v56 Optuna ensemble search')
    parser.add_argument('--n_trials',    type=int,   default=300,
                        help='Optuna trials per study (default 300)')
    parser.add_argument('--w_max',       type=float, default=1.2,
                        help='Upper bound for blend weights (default 1.2)')
    parser.add_argument('--allow_q3s3', action='store_true',
                        help='Allow Q3/S3 weights > 0 (default: locked to 0)')
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    train    = pd.read_csv(TRAIN_PATH)
    base_oof, base_sub = load_pair(BASE_TAG)
    mid_oof,  mid_sub  = load_pair(MID_TAG)
    best_oof, best_sub = load_pair(BEST_TAG)

    results = []

    # ── A: 2-way Optuna ───────────────────────────────────────────────────────
    print(f'[v56] 2-way Optuna search  (n_trials={args.n_trials}) …')
    sampler_2 = optuna.samplers.TPESampler(seed=args.seed)
    study_2   = optuna.create_study(direction='minimize', sampler=sampler_2)
    study_2.optimize(
        make_objective_2way(train, base_oof, mid_oof, args.allow_q3s3, args.w_max),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )
    best2 = study_2.best_trial
    alphas_2 = {t: best2.params.get(f'alpha_{t}', 0.0) for t in TARGETS}

    oof_2 = blend_2way(base_oof, mid_oof, alphas_2)
    sub_2 = blend_2way(base_sub, mid_sub, alphas_2)
    res2  = save_result(
        'v56_optuna_2way', train, oof_2, sub_2, base_oof, base_sub,
        alphas_2,
        f'Optuna 2-way (BASE→MID), n_trials={args.n_trials}, w_max={args.w_max}',
    )
    results.append(res2)
    print(f'  2-way best  proxy={best2.value:.6f} → saved')
    print(f'  alphas: { {t: f"{alphas_2[t]:.3f}" for t in TARGETS} }')

    # ── B: 3-way Optuna ───────────────────────────────────────────────────────
    print(f'[v56] 3-way Optuna search  (n_trials={args.n_trials}) …')
    sampler_3 = optuna.samplers.TPESampler(seed=args.seed)
    study_3   = optuna.create_study(direction='minimize', sampler=sampler_3)
    study_3.optimize(
        make_objective_3way(train, base_oof, mid_oof, best_oof,
                            args.allow_q3s3, args.w_max),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )
    best3    = study_3.best_trial
    alphas_3 = {t: best3.params.get(f'alpha_{t}', 0.0) for t in TARGETS}
    betas_3  = {t: best3.params.get(f'beta_{t}',  0.0) for t in TARGETS}

    oof_3 = blend_3way(base_oof, mid_oof, best_oof, alphas_3, betas_3)
    sub_3 = blend_3way(base_sub, mid_sub, best_sub, alphas_3, betas_3)
    weights_3 = {t: {'alpha': alphas_3[t], 'beta': betas_3[t]} for t in TARGETS}
    res3  = save_result(
        'v56_optuna_3way', train, oof_3, sub_3, base_oof, base_sub,
        weights_3,
        f'Optuna 3-way (BASE+MID+BEST), n_trials={args.n_trials}, w_max={args.w_max}',
    )
    results.append(res3)
    print(f'  3-way best  proxy={best3.value:.6f} → saved')
    print(f'  alphas(→MID):  { {t: f"{alphas_3[t]:.3f}" for t in TARGETS} }')
    print(f'  betas (→BEST): { {t: f"{betas_3[t]:.3f}" for t in TARGETS} }')

    # ── also save a few interpretable fixed configs for sanity ────────────────
    FIXED_SPECS = [
        (
            'v56_no_q3s3_uniform110',
            {'Q1': 1.1, 'Q2': 1.1, 'Q3': 0.0, 'S1': 1.1, 'S2': 1.1, 'S3': 0.0, 'S4': 1.1},
            'Uniform 1.1 scale on non-harmful targets (sanity anchor).',
        ),
        (
            'v56_q2_probe_zero',
            {'Q1': 1.0, 'Q2': 0.0, 'Q3': 0.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.0},
            'Drop Q2 entirely; isolate effect of Q2 removal from best.',
        ),
    ]
    for name, alphas_f, note in FIXED_SPECS:
        oof_f = blend_2way(base_oof, mid_oof, alphas_f)
        sub_f = blend_2way(base_sub, mid_sub, alphas_f)
        res_f = save_result(name, train, oof_f, sub_f, base_oof, base_sub, alphas_f, note)
        results.append(res_f)
        loss_f, _ = evaluate(train, oof_f)
        print(f'  fixed {name}: proxy={loss_f:.6f}')

    # ── summary ───────────────────────────────────────────────────────────────
    ranked = sorted(results, key=lambda x: x['oof_proxy_loss'])

    # Per-target comparison table
    per_target_table = {}
    _, base_per = evaluate(train, base_oof)
    for r in ranked:
        per_target_table[r['name']] = {
            t: round(r['oof_proxy_per_target'][t] - base_per[t], 6)
            for t in TARGETS
        }

    summary = {
        'exp_tag':             'v56_optuna_3way',
        'optuna_config':       vars(args),
        'known_public_scores': PUBLIC_SCORES,
        'base_oof_per_target': base_per,
        'per_target_delta_vs_base': per_target_table,
        'recommended_submit_order': [r['name'] for r in ranked[:3]],
        'candidates':          ranked,
    }
    path = SUMMARY_DIR / 'summary_v56_optuna_3way.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'\n[v56] summary → {path}')
    print('  Ranked by OOF proxy loss:')
    for r in ranked:
        delta = r['oof_proxy_loss'] - PUBLIC_SCORES[BASE_TAG]
        sign  = '+' if delta >= 0 else ''
        print(
            f"    {r['name']}: "
            f"proxy={r['oof_proxy_loss']:.6f}  "
            f"Δbase={sign}{delta:.6f}  "
            f"mad={r['distribution_vs_base']['mad_vs_base']:.6f}"
        )
    print(f'\n  → Submit first: {ranked[0]["name"]}')
    print(f"    submission: {ranked[0]['submission']}")


if __name__ == '__main__':
    main()
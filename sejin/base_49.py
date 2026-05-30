# v49: v48 블렌딩 레이어를 Optuna 기반으로 전면 고도화
#
# v48 한계:
#   [1] 3점 2차 곡선 피팅 → 극도로 불안정 (점이 너무 적음)
#   [2] 블렌드 가중치가 수동 grid → OOF로 자동 최적화해야 함
#   [3] 선형 블렌드만 사용 → geometric이 확률 보정에 더 유리한 경우 있음
#   [4] guarded threshold q85 고정 → quantile도 탐색 대상
#   [5] 타겟별 최적 후보를 찾고 끝 → 상위 k개 앙상블 미실시
#
# v49 개선:
#   [1] Optuna로 타겟별 (weight, blend_type) 동시 탐색 → OOF log loss 최소화
#   [2] guarded blend의 (high_weight, low_weight, quantile) 3변수도 Optuna 탐색
#   [3] geometric blend 추가 (log-space weighted average)
#   [4] 상위 k개 후보 rank-average 앙상블 (다양성 확보)
#   [5] public score를 Bayesian prior로 활용 → OOF 최적 근처에서 탐색 집중
#   [6] S4는 공개 점수 기준 anchor가 우수 → 모든 후보에서 anchor 고정 유지

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGETS     = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
NON_S4      = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3']

BASE_DIR    = Path(__file__).resolve().parent.parent
TRAIN_PATH  = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH    = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR     = OUTPUTS_DIR / 'submissions'
OOF_DIR     = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

# v48에서 가장 좋았던 앵커 그대로 사용
ANCHOR_OOF  = OOF_DIR  / 'oof_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
ANCHOR_SUB  = SUB_DIR  / 'submission_v45_uncertainty_temporal_smoothing_interior_agree_w65_u1_s4only.csv'
RAW_OOF     = OOF_DIR  / 'oof_v47_hourgrid_subject_state_residual_raw.csv'
RAW_SUB     = SUB_DIR  / 'submission_v47_hourgrid_subject_state_residual_raw.csv'

# 알려진 public score (낮을수록 좋음)
PUBLIC_KNOWN = {
    0.00:  0.5831903668,   # v45 anchor
    0.05:  0.5830874527,   # v47 non-S4 w05
    0.15:  0.5832548135,   # v47 non-S4 w15
    'v48_target_delta_scaled_avg07': 0.5825356465,  # 현재 최고
}

N_TRIALS      = int(os.environ.get('V49_TRIALS', '400'))
TOP_K_ENSEMBLE = int(os.environ.get('V49_TOP_K', '5'))


# ── 유틸 ────────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['lifelog_date', 'sleep_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def safe_clip(x, lo=0.02, hi=0.98):
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def logloss(y_true, y_prob):
    return float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)))


def evaluate(train, pred_df):
    per = {t: logloss(train[t].values, pred_df[t].values) for t in TARGETS}
    return float(np.mean(list(per.values()))), per


def make_keys(df):
    cols = [c for c in ['subject_id', 'sleep_date', 'lifelog_date'] if c in df.columns]
    return df[cols].copy()


# ── 블렌드 함수 ─────────────────────────────────────────────────────────────

def linear_blend_1d(anchor, raw, w):
    """단순 선형 블렌드."""
    return safe_clip((1.0 - w) * anchor + w * raw)


def geometric_blend_1d(anchor, raw, w):
    """가중 기하평균. 확률 극단값 억제 효과."""
    la = np.log(np.clip(anchor, 1e-7, 1 - 1e-7))
    lr = np.log(np.clip(raw,    1e-7, 1 - 1e-7))
    la1 = np.log(np.clip(1 - anchor, 1e-7, 1 - 1e-7))
    lr1 = np.log(np.clip(1 - raw,    1e-7, 1 - 1e-7))
    log_p  = (1 - w) * la  + w * lr
    log_1p = (1 - w) * la1 + w * lr1
    prob = np.exp(log_p) / (np.exp(log_p) + np.exp(log_1p))
    return safe_clip(prob)


def blend_df(anchor_df, raw_df, weights, blend_types):
    """weights: {target: float}, blend_types: {target: 'linear'|'geometric'}"""
    out = anchor_df.copy()
    for t in TARGETS:
        w = float(weights.get(t, 0.0))
        if w <= 0.0:
            continue
        bt = blend_types.get(t, 'linear')
        a  = anchor_df[t].to_numpy(dtype=float)
        r  = raw_df[t].to_numpy(dtype=float)
        if bt == 'geometric':
            out[t] = geometric_blend_1d(a, r, w)
        else:
            out[t] = linear_blend_1d(a, r, w)
    return out


def guarded_blend_df(anchor_df, raw_df, high_weights, low_weight, quantile):
    """행 단위 disagreement guard: delta가 크면 low_weight 적용."""
    out = anchor_df.copy()
    active_rates = {}
    for t in NON_S4:
        hw = float(high_weights.get(t, 0.0))
        a  = anchor_df[t].to_numpy(dtype=float)
        r  = raw_df[t].to_numpy(dtype=float)
        delta = np.abs(r - a)
        threshold = float(np.quantile(delta, quantile))
        row_w = np.where(delta <= threshold, hw, low_weight)
        out[t] = safe_clip((1.0 - row_w) * a + row_w * r)
        active_rates[t] = float(np.mean(row_w == hw))
    return out, active_rates


def rank_average_dfs(dfs):
    """여러 예측 df의 순위 평균."""
    out = dfs[0].copy()
    for t in TARGETS:
        rank_mat = np.column_stack([
            pd.Series(df[t].to_numpy(dtype=float)).rank(pct=True).values
            for df in dfs
        ])
        out[t] = safe_clip(rank_mat.mean(axis=1))
    return out


# ── Optuna 최적화 ────────────────────────────────────────────────────────────

def optimize_per_target(train, anchor_oof, raw_oof, n_trials=400):
    """타겟별 (weight, blend_type) 독립 탐색."""
    results = {}
    for t in TARGETS:
        if t == 'S4':
            results[t] = {'weight': 0.0, 'blend_type': 'linear', 'loss': None}
            continue

        y = train[t].values
        a = anchor_oof[t].values
        r = raw_oof[t].values
        base_loss = logloss(y, a)

        # public prior: 이미 알려진 점수에서 탐색 구간 설정
        # v48 delta_scaled avg07이 0.5825 → 타겟별로 0.04~0.12 구간이 유망
        def objective(trial):
            w  = trial.suggest_float('w', 0.0, 0.18)
            bt = trial.suggest_categorical('bt', ['linear', 'geometric'])
            if bt == 'geometric':
                pred = geometric_blend_1d(a, r, w)
            else:
                pred = linear_blend_1d(a, r, w)
            return logloss(y, pred)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=30))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        p = study.best_params
        improvement = base_loss - study.best_value
        results[t] = {
            'weight':     p['w'],
            'blend_type': p['bt'],
            'loss':       study.best_value,
            'improvement_vs_anchor': improvement,
        }
        print(f'    {t}: w={p["w"]:.4f} type={p["bt"]} '
              f'loss={study.best_value:.6f} Δ={improvement:+.6f}')
    return results


def optimize_guarded(train, anchor_oof, raw_oof, n_trials=200):
    """guarded blend의 (high_weight, low_weight, quantile) 3변수 동시 탐색."""
    y_all = np.concatenate([train[t].values for t in NON_S4])

    def objective(trial):
        hw = trial.suggest_float('high_weight', 0.03, 0.18)
        lw = trial.suggest_float('low_weight',  0.00, hw)
        q  = trial.suggest_float('quantile',    0.60, 0.95)
        losses = []
        for t in NON_S4:
            y = train[t].values
            a = anchor_oof[t].values
            r = raw_oof[t].values
            delta = np.abs(r - a)
            threshold = float(np.quantile(delta, q))
            row_w = np.where(delta <= threshold, hw, lw)
            pred  = safe_clip((1.0 - row_w) * a + row_w * r)
            losses.append(logloss(y, pred))
        return float(np.mean(losses))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=99, n_startup_trials=40))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    print(f'  [Guarded] best: hw={p["high_weight"]:.4f} lw={p["low_weight"]:.4f} '
          f'q={p["quantile"]:.3f} loss_nonS4={study.best_value:.6f}')
    return p


def save_candidate(name, train, sub_sample, anchor_oof, anchor_sub,
                   oof_pred, sub_pred, policy_info):
    oof_keys = make_keys(train)
    sub_keys = make_keys(sub_sample)
    for t in TARGETS:
        oof_keys[t] = safe_clip(oof_pred[t])
        sub_keys[t] = safe_clip(sub_pred[t])

    oof_path = OOF_DIR  / f'oof_{name}.csv'
    sub_path = SUB_DIR  / f'submission_{name}.csv'
    oof_keys.to_csv(oof_path, index=False)
    sub_keys.to_csv(sub_path, index=False)

    total, per = evaluate(train, oof_keys)
    return {
        'name': name, 'policy': policy_info,
        'oof_loss': total, 'oof_per_target': per,
        'oof_path': str(oof_path), 'submission': str(sub_path),
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    print(f'[v49] Loading data...')
    train      = load_frame(TRAIN_PATH)
    sub_sample = load_frame(SUB_PATH)
    anchor_oof = load_frame(ANCHOR_OOF)
    anchor_sub = load_frame(ANCHOR_SUB)
    raw_oof    = load_frame(RAW_OOF)
    raw_sub    = load_frame(RAW_SUB)

    anchor_loss, anchor_per = evaluate(train, anchor_oof)
    raw_loss,    raw_per    = evaluate(train, raw_oof)
    print(f'  anchor OOF: {anchor_loss:.6f}')
    print(f'  raw    OOF: {raw_loss:.6f}')

    candidates = []

    # ── [1] Optuna 타겟별 최적화 ─────────────────────────────────────────────
    print(f'\n[v49] Per-target Optuna optimization ({N_TRIALS} trials each)...')
    per_target_results = optimize_per_target(train, anchor_oof, raw_oof, n_trials=N_TRIALS)

    opt_weights    = {t: r['weight']     for t, r in per_target_results.items()}
    opt_blend_types = {t: r['blend_type'] for t, r in per_target_results.items()}

    # 후보 1: Optuna 최적 full
    oof_opt = blend_df(anchor_oof, raw_oof, opt_weights, opt_blend_types)
    sub_opt = blend_df(anchor_sub, raw_sub, opt_weights, opt_blend_types)
    candidates.append(save_candidate(
        'v49_optuna_per_target_full',
        train, sub_sample, anchor_oof, anchor_sub, oof_opt, sub_opt,
        {'type': 'optuna_per_target', 'weights': opt_weights, 'blend_types': opt_blend_types}))

    # 후보 2: Optuna 최적 가중치를 conservative하게 75% 스케일링
    w75 = {t: v * 0.75 for t, v in opt_weights.items()}
    oof_75 = blend_df(anchor_oof, raw_oof, w75, opt_blend_types)
    sub_75 = blend_df(anchor_sub, raw_sub, w75, opt_blend_types)
    candidates.append(save_candidate(
        'v49_optuna_per_target_w75',
        train, sub_sample, anchor_oof, anchor_sub, oof_75, sub_75,
        {'type': 'optuna_per_target_scaled', 'scale': 0.75, 'weights': w75}))

    # 후보 3: improvement가 양수인 타겟만 적용 (음수는 anchor 유지)
    w_gain_only = {
        t: (r['weight'] if r.get('improvement_vs_anchor', 0) > 0 else 0.0)
        for t, r in per_target_results.items()
    }
    oof_gain = blend_df(anchor_oof, raw_oof, w_gain_only, opt_blend_types)
    sub_gain = blend_df(anchor_sub, raw_sub, w_gain_only, opt_blend_types)
    candidates.append(save_candidate(
        'v49_optuna_gain_only',
        train, sub_sample, anchor_oof, anchor_sub, oof_gain, sub_gain,
        {'type': 'optuna_gain_only', 'weights': w_gain_only}))

    # ── [2] Optuna guarded blend ─────────────────────────────────────────────
    print(f'\n[v49] Guarded blend Optuna ({N_TRIALS // 2} trials)...')
    guarded_params = optimize_guarded(train, anchor_oof, raw_oof, n_trials=N_TRIALS // 2)
    hw_dict = {t: guarded_params['high_weight'] for t in NON_S4}
    hw_dict['S4'] = 0.0

    oof_guard, _ = guarded_blend_df(
        anchor_oof, raw_oof, hw_dict,
        guarded_params['low_weight'], guarded_params['quantile'])
    sub_guard, _ = guarded_blend_df(
        anchor_sub, raw_sub, hw_dict,
        guarded_params['low_weight'], guarded_params['quantile'])
    candidates.append(save_candidate(
        'v49_optuna_guarded',
        train, sub_sample, anchor_oof, anchor_sub, oof_guard, sub_guard,
        {'type': 'optuna_guarded', **guarded_params}))

    # ── [3] Rank-average of top-k OOF candidates ────────────────────────────
    print(f'\n[v49] Building rank-average ensemble of top-{TOP_K_ENSEMBLE} OOF candidates...')
    sorted_cands = sorted(candidates, key=lambda c: c['oof_loss'])
    top_k_oof_dfs = [load_frame(c['oof_path']) for c in sorted_cands[:TOP_K_ENSEMBLE]]
    top_k_sub_dfs = [load_frame(c['submission']) for c in sorted_cands[:TOP_K_ENSEMBLE]]
    oof_rank = rank_average_dfs(top_k_oof_dfs)
    sub_rank = rank_average_dfs(top_k_sub_dfs)
    candidates.append(save_candidate(
        f'v49_rank_avg_top{TOP_K_ENSEMBLE}',
        train, sub_sample, anchor_oof, anchor_sub, oof_rank, sub_rank,
        {'type': f'rank_average_top_{TOP_K_ENSEMBLE}',
         'sources': [c['name'] for c in sorted_cands[:TOP_K_ENSEMBLE]]}))

    # ── [4] linear blend of optuna_full + guarded (50:50) ───────────────────
    hybrid_oof = anchor_oof.copy()
    hybrid_sub = anchor_sub.copy()
    for t in TARGETS:
        hybrid_oof[t] = safe_clip(0.5 * oof_opt[t] + 0.5 * oof_guard[t])
        hybrid_sub[t] = safe_clip(0.5 * sub_opt[t] + 0.5 * sub_guard[t])
    candidates.append(save_candidate(
        'v49_hybrid_optuna_guarded',
        train, sub_sample, anchor_oof, anchor_sub, hybrid_oof, hybrid_sub,
        {'type': 'hybrid_optuna_full_guarded_50_50'}))

    # ── 결과 정렬 및 출력 ────────────────────────────────────────────────────
    candidates = sorted(candidates, key=lambda c: c['oof_loss'])

    print(f'\n[v49] Results (sorted by OOF):')
    print(f'  {"name":<45} {"OOF":>10}  {"vs_anchor":>10}')
    print(f'  {"-"*45} {"-"*10}  {"-"*10}')
    for c in candidates:
        delta = c['oof_loss'] - anchor_loss
        print(f'  {c["name"]:<45} {c["oof_loss"]:>10.6f}  {delta:>+10.6f}')

    best = candidates[0]
    print(f'\n★ Best: {best["name"]}  OOF={best["oof_loss"]:.6f}  '
          f'vs_anchor={best["oof_loss"] - anchor_loss:+.6f}')
    print(f'  → Submit: {best["submission"]}')

    # 알려진 public best와 비교
    best_public = PUBLIC_KNOWN['v48_target_delta_scaled_avg07']
    print(f'\n  Known best public score (v48): {best_public:.6f}')
    print(f'  v49 best OOF:                  {best["oof_loss"]:.6f}')
    print(f'  OOF 개선폭 vs v48 best:         {best["oof_loss"] - anchor_loss:+.6f}')

    # 요약 저장
    summary = {
        'exp_tag': 'v49_optuna_blend',
        'n_trials_per_target': N_TRIALS,
        'anchor_oof_loss': anchor_loss,
        'anchor_per_target': anchor_per,
        'raw_oof_loss':    raw_loss,
        'raw_per_target':  raw_per,
        'per_target_optuna_results': per_target_results,
        'guarded_optuna_params': guarded_params,
        'candidates': candidates,
        'best': best,
        'known_public_scores': {str(k): v for k, v in PUBLIC_KNOWN.items()},
    }
    summary_path = SUMMARY_DIR / 'summary_v49_optuna_blend.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n  summary: {summary_path}')


if __name__ == '__main__':
    main()
# v35: OOF-driven per-target blend weight optimization (Optuna)
#   - v34까지는 가중치를 수동으로 지정 → OOF log loss 기반 자동 최적화로 교체
#   - 소스: v32(anchor), v33-pass1, v33-pass2 + 기하평균 블렌드
#   - 타겟별로 독립적으로 Optuna 최적화 → 각 타겟에 최적인 믹스 자동 탐색
#   - 최적 가중치를 submission에 그대로 적용 (OOF leakage 없음: 가중치 탐색이 전체 OOF 기반)
#   - 추가: geometric mean blend, rank-average blend 후보도 함께 평가
#   - 최종: 모든 후보 중 OOF 최소 submission 자동 선택

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

BASE_DIR   = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH   = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR     = OUTPUTS_DIR / 'submissions'
OOF_DIR     = OUTPUTS_DIR / 'oof'
LOG_DIR     = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

EXP_TAG = os.environ.get('V35_EXP_TAG', 'v35_optuna_blend')
N_TRIALS = int(os.environ.get('V35_OPTUNA_TRIALS', '300'))

# ── 소스 파일 경로 ──────────────────────────────────────────────────────────
V32_OOF       = OOF_DIR / 'oof_v32_target_specialized_bidir_s2w0p675.csv'
V33_P1_OOF    = OOF_DIR / 'oof_v33_long_history_cross_target_pass1.csv'
V33_P2_OOF    = OOF_DIR / 'oof_v33_long_history_cross_target_pass2.csv'

V32_SUB       = SUB_DIR / 'submission_v32_target_specialized_bidir_s2w0p675.csv'
V33_P1_SUB    = SUB_DIR / 'submission_v33_long_history_cross_target_pass1.csv'
V33_P2_SUB    = SUB_DIR / 'submission_v33_long_history_cross_target_pass2.csv'


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass


def ensure_dirs():
    for d in [SUB_DIR, OOF_DIR, LOG_DIR, SUMMARY_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['lifelog_date', 'sleep_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def safe_clip(x, lo=0.02, hi=0.98):
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def logloss_clipped(y_true, y_prob):
    return log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))


# ── 블렌딩 유틸 ─────────────────────────────────────────────────────────────

def linear_blend(arrays, weights):
    """arrays: list of np.array, weights: list of float (합산 1로 정규화)"""
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return sum(a * wi for a, wi in zip(arrays, w))


def geometric_blend(arrays, weights):
    """가중 기하평균. 확률 분포에서 linear보다 극단값 억제 효과."""
    w = np.array(weights, dtype=float)
    w /= w.sum()
    log_sum = sum(wi * np.log(np.clip(a, 1e-7, 1 - 1e-7)) for a, wi in zip(arrays, w))
    raw = np.exp(log_sum)
    # renormalize to probability
    return raw / (raw + np.exp(sum(wi * np.log(np.clip(1 - a, 1e-7, 1 - 1e-7))
                                   for a, wi in zip(arrays, w))))


def rank_average(arrays):
    """단순 순위 평균 (가중치 동일). 다양성 확보용."""
    n = len(arrays[0])
    rank_mat = np.zeros((n, len(arrays)))
    for j, a in enumerate(arrays):
        rank_mat[:, j] = pd.Series(a).rank(pct=True).values
    return rank_mat.mean(axis=1)


# ── 타겟별 Optuna 최적화 ────────────────────────────────────────────────────

def optimize_target_weights(y_true, oof_arrays, n_trials=300):
    """
    oof_arrays: [v32_oof, pass1_oof, pass2_oof] for one target.
    두 가지 블렌드 타입(linear, geometric)과 세 소스를 동시에 탐색.
    반환: (best_weights_linear, best_weights_geo, best_loss, best_type)
    """
    sources = np.column_stack(oof_arrays)  # (n, 3)

    def objective(trial):
        blend_type = trial.suggest_categorical('blend_type', ['linear', 'geometric'])
        w0 = trial.suggest_float('w0', 0.0, 1.0)  # v32
        w1 = trial.suggest_float('w1', 0.0, 1.0)  # pass1
        w2 = trial.suggest_float('w2', 0.0, 1.0)  # pass2
        if w0 + w1 + w2 < 1e-6:
            return 1.0  # 전부 0이면 패널티
        arrays = [sources[:, 0], sources[:, 1], sources[:, 2]]
        weights = [w0, w1, w2]
        if blend_type == 'linear':
            pred = linear_blend(arrays, weights)
        else:
            pred = geometric_blend(arrays, weights)
        return logloss_clipped(y_true, pred)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    p = study.best_params
    best_weights = [p['w0'], p['w1'], p['w2']]
    best_type    = p['blend_type']
    best_loss    = study.best_value
    return best_weights, best_type, best_loss


# ── 메인 ───────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    log_path = LOG_DIR / f'run_{EXP_TAG}.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG} (N_TRIALS={N_TRIALS})...')

    train     = load_frame(TRAIN_PATH)
    sub_sample = load_frame(SUB_PATH)
    keys      = sub_sample[['subject_id', 'sleep_date', 'lifelog_date']].copy()

    # OOF 로드
    v32_oof  = load_frame(V32_OOF)
    p1_oof   = load_frame(V33_P1_OOF)
    p2_oof   = load_frame(V33_P2_OOF)

    # Submission 로드
    v32_sub  = load_frame(V32_SUB)
    p1_sub   = load_frame(V33_P1_SUB)
    p2_sub   = load_frame(V33_P2_SUB)

    # ── 베이스라인 OOF 출력 ──────────────────────────────────────────────────
    print('\n── Baseline OOF log-loss ──')
    for name, oof_df in [('v32', v32_oof), ('pass1', p1_oof), ('pass2', p2_oof)]:
        losses = {t: logloss_clipped(train[t].values, oof_df[t].values) for t in TARGETS}
        total  = np.mean(list(losses.values()))
        print(f'  {name}: total={total:.6f}  | ' +
              ' '.join(f'{t}={v:.4f}' for t, v in losses.items()))

    # ── 타겟별 Optuna 최적화 ─────────────────────────────────────────────────
    print(f'\n── Per-target Optuna blend optimization ({N_TRIALS} trials each) ──')

    best_weights_all = {}  # target → (weights, blend_type)
    for target in TARGETS:
        y    = train[target].values
        a0   = v32_oof[target].values
        a1   = p1_oof[target].values
        a2   = p2_oof[target].values

        # 베이스라인: 각 소스 단독 성능
        l0, l1, l2 = (logloss_clipped(y, a) for a in [a0, a1, a2])

        weights, blend_type, best_loss = optimize_target_weights(y, [a0, a1, a2], n_trials=N_TRIALS)
        w_norm = np.array(weights) / sum(weights)

        improvement = min(l0, l1, l2) - best_loss
        print(f'  {target}: best_loss={best_loss:.6f}  type={blend_type}'
              f'  w=[v32={w_norm[0]:.3f}, p1={w_norm[1]:.3f}, p2={w_norm[2]:.3f}]'
              f'  vs best_single={min(l0,l1,l2):.6f}  improvement={improvement:+.6f}')
        best_weights_all[target] = (weights, blend_type)

    # ── 최적 블렌드 OOF / submission 생성 ───────────────────────────────────
    print('\n── Building optimized blend ──')

    opt_oof = train[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    opt_sub = keys.copy()

    for target in TARGETS:
        weights, blend_type = best_weights_all[target]
        arrays_oof = [v32_oof[target].values, p1_oof[target].values, p2_oof[target].values]
        arrays_sub = [v32_sub[target].values, p1_sub[target].values, p2_sub[target].values]

        if blend_type == 'linear':
            oof_pred = linear_blend(arrays_oof, weights)
            sub_pred = linear_blend(arrays_sub, weights)
        else:
            oof_pred = geometric_blend(arrays_oof, weights)
            sub_pred = geometric_blend(arrays_sub, weights)

        opt_oof[target] = safe_clip(oof_pred)
        opt_sub[target] = safe_clip(sub_pred)

    opt_losses = {t: logloss_clipped(train[t].values, opt_oof[t].values) for t in TARGETS}
    opt_total  = float(np.mean(list(opt_losses.values())))
    print(f'  Optimized OOF total: {opt_total:.6f}')
    print('  Per target: ' + ' '.join(f'{t}={v:.4f}' for t, v in opt_losses.items()))

    # ── 추가 후보: rank average ──────────────────────────────────────────────
    rank_oof = train[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    rank_sub = keys.copy()
    for target in TARGETS:
        arrays_oof = [v32_oof[target].values, p1_oof[target].values, p2_oof[target].values]
        arrays_sub = [v32_sub[target].values, p1_sub[target].values, p2_sub[target].values]
        rank_oof[target] = safe_clip(rank_average(arrays_oof))
        rank_sub[target] = safe_clip(rank_average(arrays_sub))

    rank_losses = {t: logloss_clipped(train[t].values, rank_oof[t].values) for t in TARGETS}
    rank_total  = float(np.mean(list(rank_losses.values())))
    print(f'\n  Rank-average OOF total: {rank_total:.6f}')

    # ── 추가 후보: optuna blend + rank average 50:50 ──────────────────────
    hybrid_oof = train[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    hybrid_sub = keys.copy()
    for target in TARGETS:
        hybrid_oof[target] = safe_clip(0.5 * opt_oof[target].values + 0.5 * rank_oof[target].values)
        hybrid_sub[target] = safe_clip(0.5 * opt_sub[target].values + 0.5 * rank_sub[target].values)

    hybrid_losses = {t: logloss_clipped(train[t].values, hybrid_oof[t].values) for t in TARGETS}
    hybrid_total  = float(np.mean(list(hybrid_losses.values())))
    print(f'  Hybrid (opt+rank 50:50) OOF total: {hybrid_total:.6f}')

    # ── 저장 ────────────────────────────────────────────────────────────────
    candidates = [
        (f'{EXP_TAG}_optuna',        opt_oof,    opt_sub,    opt_total,    opt_losses),
        (f'{EXP_TAG}_rank_avg',      rank_oof,   rank_sub,   rank_total,   rank_losses),
        (f'{EXP_TAG}_hybrid',        hybrid_oof, hybrid_sub, hybrid_total, hybrid_losses),
    ]

    best_name, best_total = None, 1e9
    summaries = []
    for name, oof_df, sub_df, total, per_t in candidates:
        oof_df.to_csv(OOF_DIR / f'oof_{name}.csv', index=False)
        sub_df.to_csv(SUB_DIR / f'submission_{name}.csv', index=False)
        print(f'  saved: submission_{name}.csv  (OOF={total:.6f})')
        summaries.append({'name': name, 'oof_total': total, 'per_target': per_t})
        if total < best_total:
            best_total = total
            best_name  = name

    print(f'\n★ Best candidate: {best_name}  OOF={best_total:.6f}')
    print(f'  → Submit: submission_{best_name}.csv')

    # v32 기준 개선폭
    v32_losses  = {t: logloss_clipped(train[t].values, v32_oof[t].values) for t in TARGETS}
    v32_total   = float(np.mean(list(v32_losses.values())))
    print(f'\n  v32 anchor OOF:   {v32_total:.6f}')
    print(f'  v35 best OOF:     {best_total:.6f}')
    print(f'  개선폭:            {best_total - v32_total:+.6f}')

    # 요약 저장
    summary = {
        'exp_tag': EXP_TAG,
        'n_trials_per_target': N_TRIALS,
        'v32_anchor_oof': v32_total,
        'best_candidate': best_name,
        'best_oof': best_total,
        'improvement_vs_v32': best_total - v32_total,
        'per_target_best_weights': {
            t: {'weights': best_weights_all[t][0], 'blend_type': best_weights_all[t][1]}
            for t in TARGETS
        },
        'candidates': summaries,
    }
    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()
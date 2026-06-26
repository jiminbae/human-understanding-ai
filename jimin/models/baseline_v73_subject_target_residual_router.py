"""v73: same-subject, target-specific residual scale routing.

v72 proved publicly that per-subject residual routing transfers.  v73 splits
each subject's residual scale by target while keeping the public-validated core
axes, Q3/S3 behavior, and block roles unchanged.  Optima are estimated on each
subject's test-shaped pseudo-hidden rows and shrunk toward global scale 12.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v56_block_router as v56
import baseline_v61_subject_block_label_solver as v61
import baseline_v65_safe_residual_extension as v65


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_TAG = 'v56_block_router_mid'
GLOBAL_SCALE = 12.0
SCALE_GRID = np.arange(0.0, 40.01, 1.0)


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def estimate_optima(train, roles, scale_oofs):
    optima = {}
    diagnostics = {}
    for sid in sorted(train['subject_id'].unique()):
        mask = (train['subject_id'] == sid) & (roles != 'visible')
        diagnostics[sid] = {}
        for target in TARGETS:
            losses = {
                float(scale): v56.target_logloss(
                    train.loc[mask, target], frame.loc[mask, target]
                )
                for scale, frame in scale_oofs.items()
            }
            best_scale = min(losses, key=losses.get)
            optima[(sid, target)] = float(best_scale)
            diagnostics[sid][target] = {
                'best_scale': float(best_scale),
                'best_loss': float(losses[best_scale]),
                'global12_loss': float(losses[GLOBAL_SCALE]),
                'delta_vs_global12': float(losses[best_scale] - losses[GLOBAL_SCALE]),
            }
    return optima, diagnostics


def shrunk_scale(optimum, shrink):
    return float(round(GLOBAL_SCALE + float(shrink) * (float(optimum) - GLOBAL_SCALE)))


def compose(base, frames, optima, shrink, subject_ids):
    out = base.copy()
    used = {}
    for (sid, target), optimum in optima.items():
        scale = shrunk_scale(optimum, shrink)
        mask = subject_ids == sid
        out.loc[mask, target] = frames[scale].loc[mask, target]
        used.setdefault(sid, {})[target] = scale
    return out, used


def save_candidate(name, train, anchor_sub, roles, test_roles, oof, submission, scales, shrink):
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    return {
        'name': name,
        'shrink': float(shrink),
        'subject_target_scales': scales,
        'oof_path': str(oof_path),
        'submission': str(sub_path),
        'full_oof': v56.evaluate(train, oof),
        'role_oof': v56.role_evaluations(train, oof, roles),
        'distribution_vs_anchor': v56.describe_vs_anchor(submission, anchor_sub, test_roles),
    }


def main():
    ensure_dirs()
    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_SAMPLE_PATH)
    profiles, test_roles = v56.build_test_profiles(train, sub)
    roles = v56.build_train_roles(train, profiles)
    anchor_oof = v56.load_oof(ANCHOR_TAG, train)
    anchor_sub = v56.load_submission(ANCHOR_TAG, sub)
    solver_oof = v56.load_oof('v61_label_solver_raw', train)
    solver_sub = v56.load_submission('v61_label_solver_raw', sub)
    conf_oof = v56.load_oof('v61_label_solver_confidence', train)
    conf_sub = v56.load_submission('v61_label_solver_confidence', sub)

    scale_oofs = {}
    scale_subs = {}
    for scale in SCALE_GRID:
        weights = v65.scale_core_plus_residual(1.0, float(scale))
        scale_oofs[float(scale)] = v61.apply_label_bridge(
            anchor_oof, solver_oof, conf_oof, roles, weights, cap=0.16
        )
        scale_subs[float(scale)] = v61.apply_label_bridge(
            anchor_sub, solver_sub, conf_sub, test_roles, weights, cap=0.16
        )

    optima, optimum_diagnostics = estimate_optima(train, roles, scale_oofs)
    base_oof = scale_oofs[GLOBAL_SCALE]
    base_sub = scale_subs[GLOBAL_SCALE]
    candidates = []
    for shrink in [0.50, 0.75, 1.00]:
        oof, scales = compose(base_oof, scale_oofs, optima, shrink, train['subject_id'])
        submission, _ = compose(base_sub, scale_subs, optima, shrink, sub['subject_id'])
        tag = str(shrink).replace('.', 'p')
        candidates.append(save_candidate(
            f'v73_subject_target_scale_shrink{tag}',
            train,
            anchor_sub,
            roles,
            test_roles,
            oof,
            submission,
            scales,
            shrink,
        ))

    global_eval = {
        'full_oof': v56.evaluate(train, base_oof),
        'role_oof': v56.role_evaluations(train, base_oof, roles),
        'known_public_score': 0.5769907858,
    }
    summary = {
        'exp_tag': 'v73_subject_target_residual_router',
        'global_scale12': global_eval,
        'scale_grid': SCALE_GRID.tolist(),
        'subject_target_optimum_diagnostics': optimum_diagnostics,
        'candidates': candidates,
        'recommended_submit_order': [
            'v73_subject_target_scale_shrink0p75',
            'v73_subject_target_scale_shrink1p0',
            'v73_subject_target_scale_shrink0p5',
        ],
        'policy_notes': [
            'v72 public feedback validates same-subject routing.',
            'Target splitting is more aggressive and should start at shrink 0.75 under the remaining budget.',
            'Q3/S3 remain unchanged because the residual source has zero weights on those targets.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v73_subject_target_residual_router.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v73] summary={summary_path}')
    global_routed = global_eval['role_oof']['routed_rows']['loss']
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        print(
            f"  {item['name']}: full_oof={item['full_oof']['loss']:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_vs_global12={routed - global_routed:+.6f} "
            f"mad={item['distribution_vs_anchor']['mad_vs_anchor']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

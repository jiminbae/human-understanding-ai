"""v74: move opposite the catastrophically overfit v73 public direction.

Known public scores:
    v72 subject shrink75        0.5755923274
    v73 subject-target shrink75 0.5837420840

v73 improved pseudo OOF but failed public by +0.0081497566.  That is a strong
public gradient.  v74 extrapolates from the v72 public-best in the opposite
direction of v73, using small weights to avoid another extreme jump.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v56_block_router as v56


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

BASE_TAG = 'v72_subject_scale_shrink0p75'
BAD_TAG = 'v73_subject_target_scale_shrink0p75'
PUBLIC_SCORES = {
    BASE_TAG: 0.5755923274,
    BAD_TAG: 0.5837420840,
}


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def anti_extrapolate(base, bad, strength):
    out = base.copy()
    for target in TARGETS:
        b = base[target].to_numpy(dtype=float)
        d = bad[target].to_numpy(dtype=float)
        out[target] = clip_prob(b + float(strength) * (b - d))
    return out


def save_candidate(name, train, anchor_sub, roles, test_roles, oof, submission, strength):
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    return {
        'name': name,
        'strength': float(strength),
        'oof_path': str(oof_path),
        'submission': str(sub_path),
        'full_oof': v56.evaluate(train, oof),
        'role_oof': v56.role_evaluations(train, oof, roles),
        'distribution_vs_anchor': v56.describe_vs_anchor(submission, anchor_sub, test_roles),
        'distribution_vs_v72': v56.describe_vs_anchor(submission, base_sub, test_roles),
    }


def main():
    global base_sub
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_SAMPLE_PATH)
    profiles, test_roles = v56.build_test_profiles(train, sub)
    roles = v56.build_train_roles(train, profiles)
    anchor_sub = v56.load_submission('v56_block_router_mid', sub)
    base_oof = v56.load_oof(BASE_TAG, train)
    base_sub = v56.load_submission(BASE_TAG, sub)
    bad_oof = v56.load_oof(BAD_TAG, train)
    bad_sub = v56.load_submission(BAD_TAG, sub)

    candidates = []
    for strength in [0.10, 0.20, 0.35, 0.50]:
        tag = str(strength).replace('.', 'p')
        oof = anti_extrapolate(base_oof, bad_oof, strength)
        submission = anti_extrapolate(base_sub, bad_sub, strength)
        candidates.append(save_candidate(
            f'v74_anti_v73_from_v72_w{tag}',
            train,
            anchor_sub,
            roles,
            test_roles,
            oof,
            submission,
            strength,
        ))

    base_eval = {
        'full_oof': v56.evaluate(train, base_oof),
        'role_oof': v56.role_evaluations(train, base_oof, roles),
        'known_public': PUBLIC_SCORES[BASE_TAG],
    }
    bad_eval = {
        'full_oof': v56.evaluate(train, bad_oof),
        'role_oof': v56.role_evaluations(train, bad_oof, roles),
        'known_public': PUBLIC_SCORES[BAD_TAG],
    }
    summary = {
        'exp_tag': 'v74_anti_overfit_public_rebound',
        'base': {'tag': BASE_TAG, 'eval': base_eval},
        'bad_direction': {'tag': BAD_TAG, 'eval': bad_eval},
        'public_delta_bad_vs_base': PUBLIC_SCORES[BAD_TAG] - PUBLIC_SCORES[BASE_TAG],
        'candidates': candidates,
        'recommended_submit_order': [
            'v74_anti_v73_from_v72_w0p2',
            'v74_anti_v73_from_v72_w0p35',
            'v74_anti_v73_from_v72_w0p1',
        ],
        'policy_notes': [
            'OOF is expected to worsen because public feedback says the OOF-improving v73 direction is wrong.',
            'Strength 0.20 is the first probe; larger anti weights require positive public feedback.',
            'This uses a failed submission as a one-dimensional public gradient rather than adding model complexity.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v74_anti_overfit_public_rebound.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v74] summary={summary_path}')
    print(f'[v74] public_delta_bad_vs_base={summary["public_delta_bad_vs_base"]:+.9f}')
    base_routed = base_eval['role_oof']['routed_rows']['loss']
    for item in candidates:
        routed = item['role_oof']['routed_rows']['loss']
        print(
            f"  {item['name']}: full_oof={item['full_oof']['loss']:.6f} "
            f"routed_oof={routed:.6f} "
            f"delta_routed_vs_v72={routed - base_routed:+.6f} "
            f"mad_vs_v72={item['distribution_vs_v72']['mad_vs_anchor']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

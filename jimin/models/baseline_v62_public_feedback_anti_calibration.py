"""v62: public-feedback calibration for the v61 anti-label direction.

The first v61 anti-label submission transferred to public LB:
    v61_anti_label_oof_selected_safe public = 0.579548671
    v56_block_router_mid public = 0.5798876532

This script does not train a new model.  It builds calibrated blends along the
validated v61 anti-label axis so the next submissions can step up from the safe
probe without jumping straight to the broadest OOF winner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v56_block_router as v56


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

ANCHOR_TAG = 'v56_block_router_mid'
KNOWN_PUBLIC = {
    'v56_block_router_mid': 0.5798876532,
    'v61_anti_label_oof_selected_safe': 0.579548671,
}


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def blend_from_anchor(anchor: pd.DataFrame, source: pd.DataFrame, scale: float):
    out = anchor.copy()
    for target in TARGETS:
        out[target] = clip_prob(
            anchor[target].to_numpy(dtype=float)
            + float(scale) * (source[target].to_numpy(dtype=float) - anchor[target].to_numpy(dtype=float))
        )
    return out


def blend_two(left: pd.DataFrame, right: pd.DataFrame, weight_right: float):
    out = left.copy()
    w = float(weight_right)
    for target in TARGETS:
        out[target] = clip_prob(
            (1.0 - w) * left[target].to_numpy(dtype=float)
            + w * right[target].to_numpy(dtype=float)
        )
    return out


def save_candidate(name, train, anchor_sub, test_roles, train_roles, oof, submission, spec):
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    return {
        'name': name,
        'spec': spec,
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

    tags = {
        'safe': 'v61_anti_label_oof_selected_safe',
        'core_mid': 'v61_anti_label_core_mid',
        'core_bold': 'v61_anti_label_core_bold_no_q3s3',
        'oof_mid': 'v61_anti_label_oof_selected_mid',
        'oof_bold': 'v61_anti_label_oof_selected_bold_no_q3s3',
    }
    oofs = {name: v56.load_oof(tag, train) for name, tag in tags.items()}
    subs = {name: v56.load_submission(tag, sub) for name, tag in tags.items()}

    raw_specs = [
        ('v62_anti_core_mid_scale085', 'scale_from_anchor', {'source': 'core_mid', 'scale': 0.85}),
        ('v62_anti_core_mid_scale115', 'scale_from_anchor', {'source': 'core_mid', 'scale': 1.15}),
        ('v62_anti_core_mid_bold_blend50', 'blend_two', {'left': 'core_mid', 'right': 'core_bold', 'weight_right': 0.50}),
        ('v62_anti_core_mid_bold_blend70', 'blend_two', {'left': 'core_mid', 'right': 'core_bold', 'weight_right': 0.70}),
        ('v62_anti_core_mid_oofmid_blend50', 'blend_two', {'left': 'core_mid', 'right': 'oof_mid', 'weight_right': 0.50}),
    ]

    candidates = []
    for name, kind, spec in raw_specs:
        if kind == 'scale_from_anchor':
            source = spec['source']
            oof = blend_from_anchor(anchor_oof, oofs[source], spec['scale'])
            submission = blend_from_anchor(anchor_sub, subs[source], spec['scale'])
        elif kind == 'blend_two':
            oof = blend_two(oofs[spec['left']], oofs[spec['right']], spec['weight_right'])
            submission = blend_two(subs[spec['left']], subs[spec['right']], spec['weight_right'])
        else:
            raise ValueError(kind)
        candidates.append(save_candidate(
            name,
            train,
            anchor_sub,
            test_roles,
            train_roles,
            oof,
            submission,
            {'kind': kind, **spec},
        ))

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    safe_public_delta = KNOWN_PUBLIC['v61_anti_label_oof_selected_safe'] - KNOWN_PUBLIC[ANCHOR_TAG]
    safe_oof_delta = (
        v56.evaluate(train, oofs['safe'])['loss']
        - anchor_eval['full_oof']['loss']
    )
    transfer_ratio = float(safe_public_delta / safe_oof_delta) if safe_oof_delta else None
    for item in candidates:
        item['public_delta_projection_from_safe_ratio'] = (
            None if transfer_ratio is None
            else float((item['full_oof']['loss'] - anchor_eval['full_oof']['loss']) * transfer_ratio)
        )
        item['public_score_projection_from_safe_ratio'] = (
            None if item['public_delta_projection_from_safe_ratio'] is None
            else float(KNOWN_PUBLIC[ANCHOR_TAG] + item['public_delta_projection_from_safe_ratio'])
        )

    sorted_candidates = sorted(candidates, key=lambda item: item['full_oof']['loss'])
    summary = {
        'exp_tag': 'v62_public_feedback_anti_calibration',
        'anchor': {
            'tag': ANCHOR_TAG,
            'known_public': KNOWN_PUBLIC[ANCHOR_TAG],
            'eval': anchor_eval,
        },
        'public_feedback': {
            'safe_tag': tags['safe'],
            'safe_public': KNOWN_PUBLIC['v61_anti_label_oof_selected_safe'],
            'safe_public_delta': safe_public_delta,
            'safe_oof_delta': safe_oof_delta,
            'transfer_ratio_public_delta_over_oof_delta': transfer_ratio,
        },
        'input_tags': tags,
        'candidates': candidates,
        'recommended_submit_order': [
            'v61_anti_label_core_mid',
            'v62_anti_core_mid_bold_blend50',
            'v61_anti_label_core_bold_no_q3s3',
        ],
        'oof_sorted': [item['name'] for item in sorted_candidates],
        'policy_notes': [
            'Safe public feedback validated the v61 anti-label sign.',
            'Core candidates avoid the broad OOF-selected anti-label movement across many small-gain targets.',
            'The projection is only a rough calibration from one public point; use it for ordering, not as a score forecast.',
        ],
    }
    summary_path = SUMMARY_DIR / 'summary_v62_public_feedback_anti_calibration.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v62] summary={summary_path}')
    print(f'[v62] safe_public_delta={safe_public_delta:+.9f} '
          f'safe_oof_delta={safe_oof_delta:+.9f} transfer_ratio={transfer_ratio:.3f}')
    anchor_routed = anchor_eval['role_oof']['routed_rows']['loss']
    for item in candidates:
        print(
            f"  {item['name']}: full_oof={item['full_oof']['loss']:.6f} "
            f"routed_oof={item['role_oof']['routed_rows']['loss']:.6f} "
            f"delta_routed={item['role_oof']['routed_rows']['loss'] - anchor_routed:+.6f} "
            f"mad={item['distribution_vs_anchor']['mad_vs_anchor']:.6f} "
            f"proj={item['public_score_projection_from_safe_ratio']:.6f} "
            f"sub={item['submission']}"
        )


if __name__ == '__main__':
    main()

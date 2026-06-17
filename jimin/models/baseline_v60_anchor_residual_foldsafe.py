from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import baseline_v56_block_router as v56


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
RESIDUAL_TARGETS = ['Q2', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_SAMPLE_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'
FEATURE_CACHE = OUTPUTS_DIR / 'features' / 'features_v47_hourgrid_subject_state_residual.pkl'

ANCHOR_TAG = 'v56_block_router_mid'
N_FOLDS = 5
RIDGE_ALPHA = 80.0
TOP_FEATURES = 260
RESIDUAL_CAP = 0.20

EXCLUDED_FEATURE_TOKENS = [
    'subject_n_rows_all',
    'subject_order',
    'subject_pos_frac',
    'gap_prev_sleep',
    'gap_next_sleep',
]


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def valid_feature(name: str) -> bool:
    lower = name.lower()
    return not any(token in lower for token in EXCLUDED_FEATURE_TOKENS)


def load_cached_features():
    cached = pd.read_pickle(FEATURE_CACHE)
    train_full = cached['train_full'].reset_index(drop=True)
    test_full = cached['test_full'].reset_index(drop=True)
    feature_cols = [col for col in cached['feature_cols'] if valid_feature(col)]
    return train_full, test_full, feature_cols, cached.get('metadata', {})


def align_prediction(tag: str, train: pd.DataFrame, sub: pd.DataFrame):
    oof = v56.load_oof(tag, train)
    submission = v56.load_submission(tag, sub)
    return oof, submission


def build_relative_block_folds(train: pd.DataFrame, n_folds: int):
    fold_id = pd.Series(-1, index=train.index, dtype=int)
    for _, grp in train.groupby('subject_id', sort=True):
        idx = grp.sort_values('sleep_date').index.to_numpy()
        assignments = np.floor(np.arange(len(idx)) * n_folds / len(idx)).astype(int)
        assignments = np.clip(assignments, 0, n_folds - 1)
        fold_id.loc[idx] = assignments

    folds = []
    for fold in range(n_folds):
        val_idx = np.flatnonzero(fold_id.to_numpy() == fold)
        tr_idx = np.flatnonzero(fold_id.to_numpy() != fold)
        folds.append((tr_idx, val_idx))
    return folds, fold_id


def add_meta_features(frame: pd.DataFrame, roles: pd.Series, anchor: pd.DataFrame):
    meta = pd.DataFrame(index=frame.index)
    meta['role_simple_interior'] = (roles == 'simple_interior').astype(float).to_numpy()
    meta['role_fragmented_interior'] = (roles == 'fragmented_interior').astype(float).to_numpy()
    meta['role_tail'] = (roles == 'tail').astype(float).to_numpy()
    meta['role_visible'] = (roles == 'visible').astype(float).to_numpy()
    for target in RESIDUAL_TARGETS:
        p = np.clip(anchor[target].to_numpy(dtype=float), 0.02, 0.98)
        meta[f'anchor_{target}'] = p
        meta[f'anchor_{target}_logit'] = np.log(p / (1.0 - p))

    sid_dummies = pd.get_dummies(frame['subject_id'], prefix='sid', dtype=float)
    return pd.concat([meta, sid_dummies.reset_index(drop=True)], axis=1)


def build_feature_frames(
    train_full: pd.DataFrame,
    test_full: pd.DataFrame,
    feature_cols: list[str],
    train_roles: pd.Series,
    test_roles: pd.Series,
    anchor_oof: pd.DataFrame,
    anchor_sub: pd.DataFrame,
):
    numeric_cols = []
    for col in feature_cols:
        if col in train_full.columns and col in test_full.columns and pd.api.types.is_numeric_dtype(train_full[col]):
            numeric_cols.append(col)

    x_train = train_full[numeric_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    x_test = test_full[numeric_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    x_train = pd.concat(
        [x_train, add_meta_features(train_full, train_roles.reset_index(drop=True), anchor_oof)],
        axis=1,
    )
    x_test = pd.concat(
        [x_test, add_meta_features(test_full, test_roles.reset_index(drop=True), anchor_sub)],
        axis=1,
    )
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
    return x_train, x_test


def select_top_features(x_train_imp: np.ndarray, y_train: np.ndarray, top_n: int):
    y = y_train - float(np.mean(y_train))
    x = x_train_imp - np.mean(x_train_imp, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(x * x, axis=0)) * max(np.sqrt(float(np.sum(y * y))), 1e-12)
    scores = np.abs(np.sum(x * y[:, None], axis=0) / np.maximum(denom, 1e-12))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    n = min(top_n, x_train_imp.shape[1])
    if n == x_train_imp.shape[1]:
        return np.arange(x_train_imp.shape[1])
    return np.argsort(scores)[-n:]


def fit_residual_models(
    train: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    anchor_oof: pd.DataFrame,
):
    folds, fold_id = build_relative_block_folds(train, N_FOLDS)
    pred_oof = pd.DataFrame(0.0, index=train.index, columns=RESIDUAL_TARGETS)
    pred_test = pd.DataFrame(0.0, index=x_test.index, columns=RESIDUAL_TARGETS)
    diagnostics = {}

    for target in RESIDUAL_TARGETS:
        residual = train[target].to_numpy(dtype=float) - anchor_oof[target].to_numpy(dtype=float)
        fold_scores = []
        test_accum = np.zeros(len(x_test), dtype=float)

        for fold, (tr_idx, val_idx) in enumerate(folds):
            imputer = SimpleImputer(strategy='median')
            x_tr_imp = imputer.fit_transform(x_train.iloc[tr_idx])
            x_val_imp = imputer.transform(x_train.iloc[val_idx])
            x_test_imp = imputer.transform(x_test)

            top_idx = select_top_features(x_tr_imp, residual[tr_idx], TOP_FEATURES)
            scaler = StandardScaler()
            x_tr_sel = scaler.fit_transform(x_tr_imp[:, top_idx])
            x_val_sel = scaler.transform(x_val_imp[:, top_idx])
            x_test_sel = scaler.transform(x_test_imp[:, top_idx])

            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(x_tr_sel, residual[tr_idx])
            val_pred = np.clip(model.predict(x_val_sel), -RESIDUAL_CAP, RESIDUAL_CAP)
            test_pred = np.clip(model.predict(x_test_sel), -RESIDUAL_CAP, RESIDUAL_CAP)
            pred_oof.loc[val_idx, target] = val_pred
            test_accum += test_pred / N_FOLDS

            base_loss = v56.target_logloss(train.loc[val_idx, target], anchor_oof.loc[val_idx, target])
            corrected = np.clip(anchor_oof.loc[val_idx, target].to_numpy(dtype=float) + val_pred, 0.02, 0.98)
            corrected_loss = v56.target_logloss(train.loc[val_idx, target], corrected)
            fold_scores.append({
                'fold': int(fold),
                'n_val': int(len(val_idx)),
                'base_loss': float(base_loss),
                'corrected_loss': float(corrected_loss),
                'delta': float(corrected_loss - base_loss),
                'pred_abs_mean': float(np.mean(np.abs(val_pred))),
            })

        pred_test[target] = np.clip(test_accum, -RESIDUAL_CAP, RESIDUAL_CAP)
        diagnostics[target] = {
            'fold_scores': fold_scores,
            'oof_residual_abs_mean': float(np.mean(np.abs(pred_oof[target]))),
            'test_residual_abs_mean': float(np.mean(np.abs(pred_test[target]))),
        }

    return pred_oof, pred_test, diagnostics, fold_id


def apply_residuals(anchor: pd.DataFrame, residuals: pd.DataFrame, roles: pd.Series, spec: dict):
    out = anchor.copy()
    scale = float(spec['scale'])
    mode = spec['mode']
    for target in RESIDUAL_TARGETS:
        if target not in spec['targets']:
            continue
        if mode == 'all_rows':
            mask = pd.Series(True, index=roles.index)
        elif mode == 'all_hidden':
            mask = roles != 'visible'
        elif mode == 'public_axis':
            if target == 'Q2':
                mask = roles == 'fragmented_interior'
            else:
                mask = roles != 'visible'
        elif mode == 's4_hidden':
            mask = roles != 'visible' if target == 'S4' else pd.Series(False, index=roles.index)
        elif mode == 'q2_fragmented':
            mask = roles == 'fragmented_interior' if target == 'Q2' else pd.Series(False, index=roles.index)
        else:
            raise ValueError(f'Unknown apply mode: {mode}')
        if bool(mask.any()):
            out.loc[mask, target] = np.clip(
                out.loc[mask, target].to_numpy(dtype=float)
                + scale * residuals.loc[mask, target].to_numpy(dtype=float),
                0.02,
                0.98,
            )
    return out


CANDIDATES = {
    'v60_residual_public_axis_w25': {
        'mode': 'public_axis',
        'scale': 0.25,
        'targets': ['Q2', 'S4'],
        'note': 'Q2 fragmented + S4 hidden residual correction at 25%.',
    },
    'v60_residual_public_axis_w50': {
        'mode': 'public_axis',
        'scale': 0.50,
        'targets': ['Q2', 'S4'],
        'note': 'Q2 fragmented + S4 hidden residual correction at 50%.',
    },
    'v60_residual_public_axis_w100': {
        'mode': 'public_axis',
        'scale': 1.00,
        'targets': ['Q2', 'S4'],
        'note': 'Q2 fragmented + S4 hidden residual correction at full strength.',
    },
    'v60_residual_all_hidden_w25': {
        'mode': 'all_hidden',
        'scale': 0.25,
        'targets': ['Q2', 'S4'],
        'note': 'Q2/S4 residual correction on all pseudo-hidden roles at 25%.',
    },
    'v60_residual_s4_hidden_w50': {
        'mode': 's4_hidden',
        'scale': 0.50,
        'targets': ['S4'],
        'note': 'S4-only hidden residual correction at 50%.',
    },
    'v60_residual_q2_fragmented_w50': {
        'mode': 'q2_fragmented',
        'scale': 0.50,
        'targets': ['Q2'],
        'note': 'Q2-only fragmented residual correction at 50%.',
    },
}


def save_candidate(
    name: str,
    train: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_sub: pd.DataFrame,
    residual_oof: pd.DataFrame,
    residual_test: pd.DataFrame,
    train_roles: pd.Series,
    test_roles: pd.Series,
    spec: dict,
):
    oof = apply_residuals(anchor_oof, residual_oof, train_roles, spec)
    submission = apply_residuals(anchor_sub, residual_test, test_roles, spec)
    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof.to_csv(oof_path, index=False)
    submission.to_csv(sub_path, index=False)
    return {
        'name': name,
        'note': spec['note'],
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
    train_full, test_full, feature_cols, metadata = load_cached_features()
    profiles, test_roles = v56.build_test_profiles(train, sub)
    train_roles = v56.build_train_roles(train, profiles)
    anchor_oof, anchor_sub = align_prediction(ANCHOR_TAG, train, sub)
    x_train, x_test = build_feature_frames(
        train_full,
        test_full,
        feature_cols,
        train_roles,
        test_roles,
        anchor_oof,
        anchor_sub,
    )

    residual_oof, residual_test, residual_diagnostics, fold_id = fit_residual_models(
        train,
        x_train,
        x_test,
        anchor_oof,
    )

    anchor_eval = {
        'full_oof': v56.evaluate(train, anchor_oof),
        'role_oof': v56.role_evaluations(train, anchor_oof, train_roles),
    }
    candidates = [
        save_candidate(
            name,
            train,
            anchor_oof,
            anchor_sub,
            residual_oof,
            residual_test,
            train_roles,
            test_roles,
            spec,
        )
        for name, spec in CANDIDATES.items()
    ]

    summary = {
        'exp_tag': 'v60_anchor_residual_foldsafe',
        'anchor': {
            'tag': ANCHOR_TAG,
            'known_public_score': 0.5798876532,
            'eval': anchor_eval,
        },
        'config': {
            'n_folds': N_FOLDS,
            'ridge_alpha': RIDGE_ALPHA,
            'top_features': TOP_FEATURES,
            'residual_cap': RESIDUAL_CAP,
            'feature_cache': str(FEATURE_CACHE),
            'n_features_after_filter': int(x_train.shape[1]),
        },
        'feature_metadata': metadata,
        'residual_diagnostics': residual_diagnostics,
        'role_counts': {
            'train_pseudo': train_roles.value_counts().astype(int).to_dict(),
            'test': test_roles.value_counts().astype(int).to_dict(),
        },
        'policy_notes': [
            'This model learns only Q2/S4 residuals around v56_mid.',
            'Residual OOF is built with subject-relative block folds.',
            'Candidates limit application to public-validated roles before trying all hidden rows.',
        ],
        'candidates': candidates,
        'recommended_submit_order': [
            'v60_residual_public_axis_w25',
            'v60_residual_s4_hidden_w50',
            'v60_residual_public_axis_w50',
            'v60_residual_all_hidden_w25',
        ],
    }
    path = SUMMARY_DIR / 'summary_v60_anchor_residual_foldsafe.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'[v60] summary={path}')
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

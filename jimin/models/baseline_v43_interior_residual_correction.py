# v43: low-capacity interior residual correction on top of the public-best w40 anchor.
#   - v42 showed that replacing interior predictions with an older source does
#     not transfer to public LB, even when pseudo-interior proxies like it.
#   - v43 keeps every tail prediction unchanged and learns only a conservative
#     interior calibration from anchor logits, visible label history, and coarse
#     block geometry.
#   - Leave-one-subject-out proxy predictions are used to keep the correction
#     honest before any submission is considered.
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jimin.analysis import pseudo_public_interior_profile_eval as interior_eval
from jimin.models import baseline_v33_long_history_cross_target as v33


TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
WINDOWS = [3, 7, 14, 30, 60]
BLEND_WEIGHTS = [0.25, 0.50, 1.00]

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUB_DIR = OUTPUTS_DIR / 'submissions'
OOF_DIR = OUTPUTS_DIR / 'oof'
LOG_DIR = OUTPUTS_DIR / 'log'
SUMMARY_DIR = OUTPUTS_DIR / 'summary'

EXP_TAG = os.environ.get('V43_EXP_TAG', 'v43_interior_residual_correction')
ANCHOR_OOF_PATH = OOF_DIR / 'oof_v38_block_role_aware_tail_conservative_w40.csv'
ANCHOR_SUB_PATH = SUB_DIR / 'submission_v38_block_role_aware_tail_conservative_w40.csv'


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def ensure_dirs():
    for path in [SUB_DIR, OOF_DIR, LOG_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_frame(path):
    df = pd.read_csv(path)
    for col in ['sleep_date', 'lifelog_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.reset_index(drop=True)


def clip_prob(values):
    return np.clip(np.asarray(values, dtype=float), 0.02, 0.98)


def logit(values):
    arr = clip_prob(values)
    return np.log(arr / (1.0 - arr))


def build_actual_interior_mask(train, sub):
    mask = np.zeros(len(sub), dtype=bool)
    for sid, grp in sub.groupby('subject_id', sort=True):
        train_dates = train.loc[train['subject_id'] == sid, 'sleep_date']
        for idx, sleep_date in grp['sleep_date'].items():
            mask[idx] = bool((train_dates > sleep_date).any())
    return pd.Series(mask, index=sub.index)


def pseudo_blocks_for_subject(train_grp, profile):
    idx = train_grp.sort_values('sleep_date').index.to_numpy()
    lengths = list(profile['interior_x_runs'])
    if not lengths:
        return []
    visible_total = len(idx) - int(sum(lengths))
    raw_gaps = profile['t_runs'][:len(lengths) + 1]
    gaps = interior_eval.proportional_gaps(visible_total, raw_gaps)

    blocks = []
    cursor = gaps[0]
    for block_i, block_len in enumerate(lengths):
        block = idx[cursor:cursor + block_len].tolist()
        blocks.append(block)
        cursor += block_len + gaps[block_i + 1]
    return blocks


def actual_blocks_for_subject(train_grp, sub_grp):
    combined = pd.concat([
        train_grp[['sleep_date']].assign(kind='T', idx=train_grp.index),
        sub_grp[['sleep_date']].assign(kind='X', idx=sub_grp.index),
    ]).sort_values('sleep_date')

    blocks = []
    current = []
    seen_future_train = False
    rows = list(combined.itertuples(index=False))
    for pos, row in enumerate(rows):
        if row.kind == 'X':
            future_has_train = any(r.kind == 'T' for r in rows[pos + 1:])
            if future_has_train:
                current.append(int(row.idx))
            elif current:
                blocks.append(current)
                current = []
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def geometry_frame(index, blocks, profile):
    geom = pd.DataFrame(index=index)
    geom['geom_block_len'] = np.nan
    geom['geom_pos'] = np.nan
    geom['geom_pos_frac'] = np.nan
    geom['geom_dist_left'] = np.nan
    geom['geom_dist_right'] = np.nan
    geom['geom_is_simple'] = float(profile['is_simple'])
    geom['geom_is_fragmented'] = float(profile['is_fragmented'])
    geom['geom_n_x_runs'] = float(profile['n_x_runs'])
    for block in blocks:
        n = len(block)
        for pos, idx in enumerate(block):
            geom.loc[idx, 'geom_block_len'] = float(n)
            geom.loc[idx, 'geom_pos'] = float(pos)
            geom.loc[idx, 'geom_pos_frac'] = float(pos / max(1, n - 1))
            geom.loc[idx, 'geom_dist_left'] = float(pos)
            geom.loc[idx, 'geom_dist_right'] = float(n - 1 - pos)
    return geom


def build_geometry_features(train, sub, profiles):
    train_parts = []
    sub_parts = []
    for sid, profile in profiles.items():
        train_grp = train.loc[train['subject_id'] == sid]
        sub_grp = sub.loc[sub['subject_id'] == sid]
        train_parts.append(geometry_frame(train_grp.index, pseudo_blocks_for_subject(train_grp, profile), profile))
        sub_parts.append(geometry_frame(sub_grp.index, actual_blocks_for_subject(train_grp, sub_grp), profile))
    return (
        pd.concat(train_parts).sort_index(),
        pd.concat(sub_parts).sort_index(),
    )


def build_target_features(
    train,
    sub,
    anchor_oof,
    anchor_sub,
    train_geom,
    sub_geom,
    all_mask,
    actual_interior_mask,
    target,
    feature_set,
):
    train_query = train.loc[all_mask, ['subject_id', 'sleep_date', 'lifelog_date']].copy()
    train_query['is_hidden'] = True
    sub_query = sub.loc[actual_interior_mask, ['subject_id', 'sleep_date', 'lifelog_date']].copy()

    visible_history = train.loc[~all_mask].copy()
    pseudo_hist_map = v33._build_subject_history(visible_history, target)
    train_hist = v33._encode_bidirectional_from_history(pseudo_hist_map, train_query, WINDOWS)
    actual_hist_map = v33._build_subject_history(train, target)
    sub_hist = v33._encode_bidirectional_from_history(actual_hist_map, sub_query, WINDOWS)

    train_feat = pd.DataFrame(index=train_query.index)
    sub_feat = pd.DataFrame(index=sub_query.index)
    train_feat['anchor_logit'] = logit(anchor_oof.loc[all_mask, target])
    sub_feat['anchor_logit'] = logit(anchor_sub.loc[actual_interior_mask, target])

    if feature_set in {'history', 'history_cross'}:
        hist_cols = [
            'te_next1',
            'te_prev_dist',
            'te_next_dist',
            'te_has_left',
            'te_has_right',
            'te_bidir_mean7',
            'te_bidir_gap7',
            'te_bidir_agree7',
            'te_bidir_mean14',
            'te_bidir_gap14',
            'te_bidir_agree14',
            'te_bidir_mean30',
            'te_bidir_gap30',
            'te_bidir_agree30',
        ]
        for col in hist_cols:
            train_feat[col] = train_hist[col].to_numpy()
            sub_feat[col] = sub_hist[col].to_numpy()
        for col in train_geom.columns:
            train_feat[col] = train_geom.loc[all_mask, col].to_numpy()
            sub_feat[col] = sub_geom.loc[actual_interior_mask, col].to_numpy()

    if feature_set == 'history_cross':
        for other in TARGETS:
            if other == target:
                continue
            train_feat[f'anchor_{other}'] = anchor_oof.loc[all_mask, other].to_numpy()
            sub_feat[f'anchor_{other}'] = anchor_sub.loc[actual_interior_mask, other].to_numpy()

    return train_feat, sub_feat


def make_model(c_value):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=c_value, max_iter=2000, solver='lbfgs')),
    ])


def fit_loso_predictions(x, y, subjects, c_value):
    pred = np.zeros(len(x), dtype=float)
    for sid in sorted(subjects.unique()):
        val_mask = subjects == sid
        tr_mask = ~val_mask
        model = make_model(c_value)
        model.fit(x.loc[tr_mask], y.loc[tr_mask])
        pred[val_mask] = model.predict_proba(x.loc[val_mask])[:, 1]
    return clip_prob(pred)


def fit_full_predictions(x_train, y_train, x_sub, c_value):
    model = make_model(c_value)
    model.fit(x_train, y_train)
    return clip_prob(model.predict_proba(x_sub)[:, 1])


def evaluate(train, pred, mask):
    per_target = {
        target: float(log_loss(
            train.loc[mask, target].values,
            np.clip(pred.loc[mask, target].values, 1e-7, 1 - 1e-7),
        ))
        for target in TARGETS
    }
    return float(np.mean(list(per_target.values()))), per_target


def describe_vs_anchor(submission, anchor):
    ref = anchor[TARGETS].to_numpy().ravel()
    arr = submission[TARGETS].to_numpy().ravel()
    return {
        'corr_vs_anchor': float(np.corrcoef(ref, arr)[0, 1]),
        'mad_vs_anchor': float(np.mean(np.abs(ref - arr))),
        'max_abs_vs_anchor': float(np.max(np.abs(ref - arr))),
        'means': {target: float(submission[target].mean()) for target in TARGETS},
    }


def save_candidate(
    name,
    train,
    keys,
    anchor_oof,
    anchor_sub,
    corrected_oof,
    corrected_sub,
    simple_mask,
    fragmented_mask,
    all_mask,
    actual_interior_mask,
    weight,
):
    oof = anchor_oof.copy()
    submission = anchor_sub.copy()
    for target in TARGETS:
        oof.loc[all_mask, target] = clip_prob(
            (1.0 - weight) * anchor_oof.loc[all_mask, target].to_numpy()
            + weight * corrected_oof[target].to_numpy()
        )
        submission.loc[actual_interior_mask, target] = clip_prob(
            (1.0 - weight) * anchor_sub.loc[actual_interior_mask, target].to_numpy()
            + weight * corrected_sub[target].to_numpy()
        )

    oof_out = pd.concat([train[['subject_id', 'sleep_date', 'lifelog_date']], oof[TARGETS]], axis=1)
    sub_out = pd.concat([keys, submission[TARGETS]], axis=1)

    all_total, all_per_target = evaluate(train, oof_out, all_mask)
    simple_total, simple_per_target = evaluate(train, oof_out, simple_mask)
    fragmented_total, fragmented_per_target = evaluate(train, oof_out, fragmented_mask)

    oof_path = OOF_DIR / f'oof_{name}.csv'
    sub_path = SUB_DIR / f'submission_{name}.csv'
    oof_out.to_csv(oof_path, index=False)
    sub_out.to_csv(sub_path, index=False)

    dist = describe_vs_anchor(sub_out, anchor_sub)
    print(
        f'\n{name}: all={all_total:.6f} '
        f'simple={simple_total:.6f} fragmented={fragmented_total:.6f}'
    )
    print(f'  all_per_target={all_per_target}')
    print(f'  simple_per_target={simple_per_target}')
    print(f'  fragmented_per_target={fragmented_per_target}')
    print(f'  dist={dist}')
    print(f'  saved={sub_path}')
    return {
        'name': name,
        'blend_weight': weight,
        'all_interior_proxy': all_total,
        'simple_interior_proxy': simple_total,
        'fragmented_interior_proxy': fragmented_total,
        'all_per_target': all_per_target,
        'simple_per_target': simple_per_target,
        'fragmented_per_target': fragmented_per_target,
        'submission': str(sub_path),
        'oof_path': str(oof_path),
        'distribution': dist,
    }


def main():
    ensure_dirs()
    log_path = LOG_DIR / f'run_{EXP_TAG}.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Starting {EXP_TAG}...')
    train = load_frame(TRAIN_PATH)
    sub = load_frame(SUB_PATH)
    keys = sub[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    anchor_oof = load_frame(ANCHOR_OOF_PATH)
    anchor_sub = load_frame(ANCHOR_SUB_PATH)

    profiles = interior_eval.build_profiles(train, sub)
    simple_mask, fragmented_mask, all_mask = interior_eval.build_interior_masks(train, profiles)
    actual_interior_mask = build_actual_interior_mask(train, sub)
    train_geom, sub_geom = build_geometry_features(train, sub, profiles)
    print(
        f'proxy rows: simple={int(simple_mask.sum())} '
        f'fragmented={int(fragmented_mask.sum())} all={int(all_mask.sum())}'
    )
    print(f'actual interior rows={int(actual_interior_mask.sum())}')

    experiments = [
        ('logit_only_c010', 'logit_only', 0.10),
        ('history_c010', 'history', 0.10),
        ('history_c025', 'history', 0.25),
        ('history_cross_c010', 'history_cross', 0.10),
    ]

    experiment_summaries = []
    for exp_name, feature_set, c_value in experiments:
        corrected_oof = pd.DataFrame(index=train.index[all_mask], columns=TARGETS, dtype=float)
        corrected_sub = pd.DataFrame(index=sub.index[actual_interior_mask], columns=TARGETS, dtype=float)
        feature_counts = {}
        for target in TARGETS:
            x_train, x_sub = build_target_features(
                train,
                sub,
                anchor_oof,
                anchor_sub,
                train_geom,
                sub_geom,
                all_mask,
                actual_interior_mask,
                target,
                feature_set,
            )
            y_train = train.loc[all_mask, target].astype(int)
            subjects = train.loc[all_mask, 'subject_id']
            corrected_oof[target] = fit_loso_predictions(x_train, y_train, subjects, c_value)
            corrected_sub[target] = fit_full_predictions(x_train, y_train, x_sub, c_value)
            feature_counts[target] = int(x_train.shape[1])

        raw_oof = anchor_oof.copy()
        raw_oof.loc[all_mask, TARGETS] = corrected_oof[TARGETS].to_numpy()
        raw_all, raw_all_per = evaluate(train, raw_oof, all_mask)
        print(
            f'\n{exp_name}: raw_corrected all={raw_all:.6f} '
            f'feature_counts={feature_counts}'
        )
        print(f'  raw_all_per_target={raw_all_per}')

        candidates = []
        for weight in BLEND_WEIGHTS:
            candidates.append(save_candidate(
                f'{EXP_TAG}_{exp_name}_w{int(weight * 100):02d}',
                train,
                keys,
                anchor_oof,
                anchor_sub,
                corrected_oof,
                corrected_sub,
                simple_mask,
                fragmented_mask,
                all_mask,
                actual_interior_mask,
                weight,
            ))
        experiment_summaries.append({
            'name': exp_name,
            'feature_set': feature_set,
            'c_value': c_value,
            'feature_counts': feature_counts,
            'raw_corrected_all_proxy': raw_all,
            'raw_corrected_all_per_target': raw_all_per,
            'candidates': candidates,
        })

    summary_path = SUMMARY_DIR / f'summary_{EXP_TAG}.json'
    summary_path.write_text(json.dumps({
        'exp_tag': EXP_TAG,
        'anchor': str(ANCHOR_SUB_PATH),
        'proxy_rows': {
            'simple': int(simple_mask.sum()),
            'fragmented': int(fragmented_mask.sum()),
            'all': int(all_mask.sum()),
        },
        'actual_interior_rows': int(actual_interior_mask.sum()),
        'experiments': experiment_summaries,
    }, indent=2), encoding='utf-8')
    print(f'\nsummary saved: {summary_path}')
    log_f.close()


if __name__ == '__main__':
    main()

"""Fixed-candidate estimators, proper scores and transparent parameter export."""
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.dataset import FEATURES


def matrix(rows):
    x = np.asarray([[r[f] for f in FEATURES] for r in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features')
    return x


def fit(kind, target, rows, protocol):
    x = matrix(rows)
    y = np.asarray([r['target_up' if target == 'direction' else 'target_return'] for r in rows])
    if kind == 'constant':
        return {'kind': kind, 'value': float((y.sum() + 1) / (len(y) + 2))}
    if kind == 'zero':
        return {'kind': kind, 'value': 0.0}
    if kind == 'mean':
        return {'kind': kind, 'value': float(y.mean())}
    if kind == 'momentum':
        return {'kind': kind}
    if kind in ('logistic', 'ridge'):
        scaler = StandardScaler().fit(x)
        model = (LogisticRegression(**protocol['logistic'], random_state=protocol['seed'])
                 if kind == 'logistic' else Ridge(**protocol['ridge']))
        model.fit(scaler.transform(x), y)
        return {'kind': kind, 'model': model, 'scaler': scaler}
    if kind == 'lightgbm':
        params = dict(protocol['lightgbm'], random_state=protocol['seed'])
        model = (lgb.LGBMClassifier(**params) if target == 'direction'
                 else lgb.LGBMRegressor(objective='regression_l1', **params))
        model.fit(x, y)
        return {'kind': kind, 'model': model}
    raise ValueError(kind)


def predict(fitted, target, rows):
    x = matrix(rows)
    kind = fitted['kind']
    if kind in ('constant', 'zero', 'mean'):
        return np.full(len(x), fitted['value'])
    if kind == 'momentum':
        return np.where(x[:, FEATURES.index('momentum_63')] > 0, .6, .4)
    model = fitted['model']
    if kind in ('logistic', 'ridge'):
        x = fitted['scaler'].transform(x)
    if kind == 'lightgbm':
        # Native prediction avoids sklearn's feature-name warning for numpy fits.
        return model.booster_.predict(x, num_threads=2)
    return model.predict_proba(x)[:, 1] if target == 'direction' else model.predict(x)


def direction_metrics(rows, p, horizon):
    y = np.asarray([r['target_up'] for r in rows], dtype=float)
    p = np.asarray(p, dtype=float)
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid direction probabilities')
    result = {'n': len(y), 'brier': float(np.mean((p - y) ** 2)),
              'log_loss': float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
              'accuracy': float(np.mean((p >= .5) == y)), 'up_rate': float(y.mean()),
              'auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None}
    # A diagnostic with non-overlapping label intervals, not a confidence interval.
    anchors = []
    previous_end = ''
    for i, row in enumerate(rows):
        if row['date'] >= previous_end:
            anchors.append(i)
            previous_end = row['label_end_date']
    result['nonoverlap_n'] = len(anchors)
    result['nonoverlap_brier'] = float(np.mean((p[anchors] - y[anchors]) ** 2))
    return result


def return_metrics(rows, values):
    y = np.asarray([r['target_return'] for r in rows], dtype=float)
    pred = np.asarray(values, dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError('Invalid return estimates')
    return {'mae': float(np.mean(np.abs(pred - y))),
            'rmse': float(np.sqrt(np.mean((pred - y) ** 2))),
            'bias': float(np.mean(pred - y)), 'actual_mean': float(y.mean())}


def split_window(rows, start, end=None, min_train=504, min_test=60):
    train = [r for r in rows if r['label_end_date'] < start]
    test = [r for r in rows if r['date'] >= start and
            (end is None or (r['date'] < end and r['label_end_date'] < end))]
    if len(train) < min_train or len(test) < min_test:
        return None
    if max(r['label_end_date'] for r in train) >= min(r['date'] for r in test):
        raise ValueError('Train labels overlap validation/test boundary')
    return train, test


def choose(scores, target, protocol):
    means = {kind: float(np.mean(values)) for kind, values in scores.items()}
    winner = min(means, key=means.get)
    simple = ['constant', 'momentum'] if target == 'direction' else ['zero', 'mean']
    baseline = min(simple, key=means.get)
    if winner not in simple:
        required = (protocol['selection']['direction_min_brier_gain'] if target == 'direction'
                    else means[baseline] * protocol['selection']['return_min_relative_gain'])
        if means[baseline] - means[winner] < required:
            winner = baseline
    return winner, means


def export_spec(fitted, folder, target):
    kind = fitted['kind']
    if kind in ('constant', 'zero', 'mean', 'momentum'):
        return {k: v for k, v in fitted.items()}
    if kind == 'lightgbm':
        name = target + '.lgb.txt'
        path = folder / name
        fitted['model'].booster_.save_model(str(path))
        return {'kind': kind, 'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    model, scaler = fitted['model'], fitted['scaler']
    return {'kind': kind, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coef': np.asarray(model.coef_).reshape(-1).tolist(),
            'intercept': float(np.asarray(model.intercept_).reshape(-1)[0])}


def save_bundle(folder, code, horizon, direction, returns, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    bundle = {'schema_version': 1, 'code': code, 'horizon': horizon,
              'features': list(FEATURES), 'direction': export_spec(direction, folder, 'direction'),
              'return': export_spec(returns, folder, 'return'), 'metadata': metadata}
    path = folder / 'model.json'
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return path

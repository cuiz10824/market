"""Three return classes; classifier-only fitting and transparent saved parameters."""
import hashlib

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from research.dataset import FEATURES
from training.engine import matrix

LABELS = ('大跌', '波动', '大涨')


def labels(returns, threshold=.05, tolerance=1e-12):
    returns = np.asarray(returns, dtype=float)
    if not np.isfinite(returns).all() or threshold <= 0 or not 0 <= tolerance < threshold:
        raise ValueError('Invalid return or threshold')
    return np.where(returns < -threshold-tolerance, 0,
                    np.where(returns > threshold+tolerance, 2, 1)).astype(int)


def align_probabilities(probabilities, classes):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1-p, p])
    if p.shape[1] != len(classes):
        raise ValueError('Classifier probability dimensions differ')
    result = np.zeros((len(p), 3))
    result[:, np.asarray(classes, dtype=int)] = p
    if not np.isfinite(result).all() or np.any(result < -1e-12) or not np.allclose(result.sum(axis=1), 1):
        raise ValueError('Invalid three-class probabilities')
    return np.clip(result, 0, 1)


def fit(kind, rows, protocol):
    y = labels([r['target_return'] for r in rows], protocol['threshold'], protocol['boundary_tolerance'])
    counts = np.bincount(y, minlength=3)
    if kind == 'neutral':
        return {'kind': 'neutral', 'probabilities': [0., 1., 0.]}
    if kind == 'constant' or len(np.unique(y)) == 1:
        return {'kind': 'constant', 'probabilities': ((counts+1)/(len(y)+3)).tolist(),
                'requested_kind': kind}
    x = matrix(rows)
    balanced = kind.endswith('_balanced')
    if kind.startswith('logistic'):
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(**protocol['logistic'], random_state=protocol['seed'],
                                   class_weight='balanced' if balanced else None)
        model.fit(scaler.transform(x), y)
        return {'kind': kind, 'model': model, 'scaler': scaler}
    if kind.startswith('lightgbm'):
        model = lgb.LGBMClassifier(**protocol['lightgbm'], random_state=protocol['seed'],
                                  class_weight='balanced' if balanced else None)
        model.fit(x, y)
        return {'kind': kind, 'model': model}
    raise ValueError('Unknown classifier kind: '+kind)


def predict(fitted, rows):
    x = matrix(rows)
    if fitted['kind'] in ('neutral', 'constant'):
        return np.tile(fitted['probabilities'], (len(rows), 1))
    if fitted['kind'].startswith('logistic'):
        p = fitted['model'].predict_proba(fitted['scaler'].transform(x))
    else:
        p = fitted['model'].booster_.predict(x, num_threads=2)
    return align_probabilities(p, fitted['model'].classes_)


def metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = align_probabilities(probabilities, [0,1,2])
    predicted = p.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0,1,2], zero_division=0)
    cm = confusion_matrix(y, predicted, labels=[0,1,2])
    clipped = np.clip(p, 1e-12, 1)
    clipped /= clipped.sum(axis=1, keepdims=True)
    per_class = [{'label': name, 'support': int(support[i]), 'precision': float(precision[i]),
                  'recall': float(recall[i]), 'f1': float(f1[i])} for i,name in enumerate(LABELS)]
    return {'n': len(y), 'accuracy': float(np.mean(predicted==y)), 'macro_f1': float(f1.mean()),
            'macro_recall': float(recall.mean()),
            'balanced_accuracy': float(recall.mean()) if np.all(support>0) else None,
            'log_loss': float(-np.mean(np.log(clipped[np.arange(len(y)), y]))),
            'brier': float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))),
            'actual_counts': np.bincount(y,minlength=3).tolist(),
            'predicted_counts': np.bincount(predicted,minlength=3).tolist(),
            'per_class': per_class, 'confusion_matrix': cm.tolist()}


def choose(folds, protocol):
    means = {kind: {'macro_f1':float(np.mean([f['scores'][kind]['macro_f1'] for f in folds])),
                    'log_loss':float(np.mean([f['scores'][kind]['log_loss'] for f in folds]))}
             for kind in protocol['classification_candidates']}
    key = lambda kind: (-means[kind]['macro_f1'], means[kind]['log_loss'])
    winner = min(means, key=key)
    simple = min(['constant','neutral'], key=key)
    if winner not in ('constant','neutral') and means[winner]['macro_f1']-means[simple]['macro_f1']<protocol['min_macro_f1_gain']:
        winner = simple
    return winner, means


def export(fitted, folder):
    kind = fitted['kind']
    if kind in ('constant','neutral'):
        return dict(fitted)
    model = fitted['model']
    if kind.startswith('lightgbm'):
        path = folder/'classification.lgb.txt'
        model.booster_.save_model(str(path))
        return {'kind':kind,'classes':model.classes_.tolist(),'file':path.name,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    scaler = fitted['scaler']
    return {'kind':kind,'classes':model.classes_.tolist(),'mean':scaler.mean_.tolist(),
            'scale':scaler.scale_.tolist(),'coef':model.coef_.tolist(),
            'intercept':model.intercept_.tolist()}

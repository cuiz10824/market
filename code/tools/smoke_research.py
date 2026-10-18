"""Exercise research adapters on real stored history; no accuracy/backtest claim."""
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))


def main():
    import numpy as np
    import pandas as pd
    from research.dataset import FEATURES, build_samples, load_history, walk_forward
    from research.models import filter_regimes, fit_direction, fit_regimes

    manifest = json.loads((ROOT / 'code/references/manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / 'code/references' / item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Changed reference file: ' + item['path'])
    calendar, prices, _ = load_history('000300')
    samples = build_samples(calendar, prices, 21, code='000300')
    train, test = next(walk_forward(samples))
    if max(r['label_end_date'] for r in train) >= test[0]['date']:
        raise ValueError('Training label leaks into test interval')
    model = fit_direction(train)
    probabilities = model.predict_proba(pd.DataFrame(test).loc[:, list(FEATURES)])
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError('Invalid classifier probabilities')
    # Separate functional fit; these state probabilities are not used as classifier
    # features or presented as out-of-sample financial results.
    closes = np.array([prices[day] for day in sorted(prices)])
    returns = np.diff(np.log(closes))[:1100]
    state_fit = fit_regimes(returns[:1000])
    filtered = np.asarray(filter_regimes(state_fit, returns))
    changed = returns.copy()
    changed[1050:] *= -2
    alternate = np.asarray(filter_regimes(state_fit, changed))
    if not np.isfinite(filtered).all() or not np.allclose(filtered.sum(axis=1), 1):
        raise ValueError('Invalid filtered state probabilities')
    if not np.allclose(filtered[:1050], alternate[:1050], rtol=1e-12, atol=1e-12):
        raise ValueError('Later observations changed past filtered probabilities')
    rejected_wrong_prefix = False
    try:
        wrong = returns.copy()
        wrong[0] += .01
        filter_regimes(state_fit, wrong)
    except ValueError:
        rejected_wrong_prefix = True
    if not rejected_wrong_prefix:
        raise ValueError('Changed training prefix was accepted')
    report = {'scope': 'Functional smoke checks, NOT performance evaluation',
              'reference_files_verified': len(manifest['files']),
              'versions': {p: importlib.metadata.version(p) for p in
                           ('numpy', 'pandas', 'scikit-learn', 'statsmodels', 'lightgbm')},
              'classifier': {'code': '000300', 'horizon': 21, 'train_rows': len(train),
                             'test_rows': len(test), 'last_training_label_end': max(r['label_end_date'] for r in train),
                             'first_test_date': test[0]['date'], 'valid_probabilities': True},
              'regimes': {'train_returns': 1000, 'filtered_returns': len(returns),
                          'converged': True, 'future_perturbation_preserves_prefix': True,
                          'wrong_training_prefix_rejected': True},
              'persistent_models_saved': False, 'accuracy_evaluated': False}
    date = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    out = ROOT / 'data/reports' / f'code_smoke_{date}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

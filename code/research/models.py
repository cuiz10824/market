"""Small project adapters. Upstream examples and licenses live in code/references/.

These are research primitives, not a validated market predictor. Keep parameter
fitting inside each training fold; state numbers are not universal bull/bear labels.
"""
from .dataset import FEATURES


def fit_direction(train_rows):
    """Fit an uncalibrated probability baseline, without using any test labels."""
    import lightgbm as lgb
    import pandas as pd

    if not train_rows or len({r['target_up'] for r in train_rows}) < 2:
        raise ValueError('Need nonempty training data containing both direction classes')
    features = pd.DataFrame(train_rows).loc[:, list(FEATURES)]
    target = [r['target_up'] for r in train_rows]
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15,
                              learning_rate=.03, min_child_samples=50,
                              random_state=42, n_jobs=2, verbosity=-1)
    # No early stopping on the final test set. Tune only on an inner purged split.
    model.fit(features, target)
    return model


def fit_regimes(train_returns, states=2):
    """Fit only training returns; use 2 states initially, rather than assert 3 cycles."""
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    values = np.asarray(train_returns, dtype=float)
    if len(values) < 252 or not np.isfinite(values).all():
        raise ValueError('Need at least 252 finite training returns')
    model = MarkovRegression(pd.Series(values * 100), k_regimes=states,
                             trend='c', switching_variance=True)
    fitted = model.fit(disp=False, maxiter=300)
    if not fitted.mle_retvals.get('converged', False):
        raise RuntimeError('State model did not converge; do not consume its probabilities')
    return fitted


def filter_regimes(fitted, returns):
    """Forward probabilities using FIXED training parameters; no re-fit or smoothing.

    Pass the original training prefix followed by any new returns. The filter is
    initialized at the same beginning; appending later data cannot change earlier
    filtered probabilities. Exclude training/test overlap outside the fit window.
    """
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    values = np.asarray(returns, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite filtering returns')
    original = np.asarray(fitted.model.endog, dtype=float)
    if len(values) < len(original) or not np.allclose(values[:len(original)] * 100, original, rtol=1e-12, atol=1e-12):
        raise ValueError('Filtering must preserve the original training prefix')
    model = MarkovRegression(pd.Series(values * 100), k_regimes=fitted.model.k_regimes,
                             trend='c', switching_variance=True)
    return model.filter(fitted.params).filtered_marginal_probabilities

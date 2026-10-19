"""Odd, strictly increasing target maps; all errors are scored after inversion."""
import numpy as np


def transform(values, spec):
    values = np.asarray(values, dtype=np.float64)
    kind = spec['kind']
    if kind == 'raw':
        result = values.copy()
    else:
        alpha = float(spec['alpha'])
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Transform alpha must be positive and finite')
        if kind == 'asinh':
            result = np.arcsinh(alpha * values) / alpha
        elif kind == 'signed_exp':
            result = np.sign(values) * np.expm1(alpha * np.abs(values)) / alpha
        else:
            raise ValueError('Unknown target transform')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite transformed target; no silent clipping')
    return result


def inverse(values, spec):
    values = np.asarray(values, dtype=np.float64)
    kind = spec['kind']
    if kind == 'raw':
        result = values.copy()
    else:
        alpha = float(spec['alpha'])
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Transform alpha must be positive and finite')
        if kind == 'asinh':
            result = np.sinh(alpha * values) / alpha
        elif kind == 'signed_exp':
            result = np.sign(values) * np.log1p(alpha * np.abs(values)) / alpha
        else:
            raise ValueError('Unknown target transform')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite inverse prediction; no silent clipping')
    return result

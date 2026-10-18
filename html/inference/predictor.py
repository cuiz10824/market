"""Load saved native/JSON parameters only. No fitting, network or scheduling."""
import hashlib
import json
from pathlib import Path

import numpy as np


class Predictor:
    def __init__(self, bundle_path):
        self.path = Path(bundle_path)
        self.bundle = json.loads(self.path.read_text(encoding='utf-8'))
        if self.bundle.get('schema_version') != 1:
            raise ValueError('Unsupported model bundle schema')
        self.features = self.bundle['features']
        self._boosters = {}
        for target in ('direction', 'return'):
            spec = self.bundle[target]
            if spec['kind'] == 'lightgbm':
                import lightgbm as lgb
                name = spec['file']
                if Path(name).name != name:
                    raise ValueError('Model filename must be local to bundle')
                path = self.path.parent / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                    raise ValueError('Model file checksum mismatch')
                self._boosters[target] = lgb.Booster(model_file=str(path))

    def predict(self, rows):
        if not rows:
            return []
        x = np.asarray([[r[name] for name in self.features] for r in rows], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Missing or nonfinite feature values')
        outputs = {}
        for target in ('direction', 'return'):
            spec = self.bundle[target]
            kind = spec['kind']
            if kind == 'lightgbm':
                values = self._boosters[target].predict(x, num_threads=2)
            elif kind in ('constant', 'zero', 'mean'):
                values = np.full(len(x), spec['value'])
            elif kind == 'momentum':
                values = np.where(x[:, self.features.index('momentum_63')] > 0, .6, .4)
            elif kind in ('logistic', 'ridge'):
                z = (x - np.asarray(spec['mean'])) / np.asarray(spec['scale'])
                values = z @ np.asarray(spec['coef']) + spec['intercept']
                if kind == 'logistic':
                    values = 1 / (1 + np.exp(-np.clip(values, -700, 700)))
            else:
                raise ValueError('Unknown model kind: ' + kind)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite model output')
            outputs[target] = values
        return [{'p_up': float(p), 'expected_price_return': float(r),
                 'code': self.bundle['code'], 'horizon': self.bundle['horizon'],
                 'research_only': True}
                for p, r in zip(outputs['direction'], outputs['return'])]

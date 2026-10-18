"""Load pooled OHLCV regressors as native models; no fitting or classifiers."""
import hashlib
import json
from pathlib import Path

import numpy as np


class EnhancedPredictor:
    def __init__(self,path):
        path=Path(path)
        self.bundle=json.loads(path.read_text(encoding='utf-8'))
        if self.bundle.get('schema_version')!=1 or self.bundle.get('task')!='enriched_return':raise ValueError('Unknown rich model')
        spec=self.bundle['return']
        if Path(spec['file']).name!=spec['file']:raise ValueError('Invalid model filename')
        native=path.parent/spec['file']
        if hashlib.sha256(native.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Enhanced model digest mismatch')
        if spec['kind'].startswith('catboost'):
            from catboost import CatBoostRegressor
            self.model=CatBoostRegressor();self.model.load_model(str(native))
        elif spec['kind'].startswith('lightgbm'):
            import lightgbm as lgb
            self.model=lgb.Booster(model_file=str(native))
        else:raise ValueError('Unknown enhanced regression kind')

    def predict(self,rows):
        x=np.asarray([[r[f] for f in self.bundle['features']] for r in rows],dtype=float)
        if not np.isfinite(x).all():raise ValueError('Nonfinite OHLCV features')
        if self.bundle['return']['kind'].startswith('lightgbm'):p=self.model.predict(x,num_threads=2)
        else:p=self.model.predict(x,thread_count=2)
        if not np.isfinite(p).all():raise ValueError('Nonfinite enhanced prediction')
        return np.asarray(p,dtype=float)

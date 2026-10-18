"""Three-class inference with the original, unchanged regression bundle."""
import hashlib
import json
from pathlib import Path

import numpy as np

from inference.predictor import Predictor

ROOT = Path(__file__).resolve().parents[2]


class ThreeClassPredictor:
    def __init__(self, path, root=ROOT):
        self.path = Path(path)
        self.bundle = json.loads(self.path.read_text(encoding='utf-8'))
        if self.bundle.get('task') != 'three_class_return' or self.bundle.get('schema_version') != 1:
            raise ValueError('Unsupported three-class bundle')
        spec = self.bundle['classifier']
        self.booster = None
        if spec['kind'].startswith('lightgbm'):
            if Path(spec['file']).name != spec['file']:
                raise ValueError('Invalid native model filename')
            model = self.path.parent/spec['file']
            if hashlib.sha256(model.read_bytes()).hexdigest()!=spec['sha256']:
                raise ValueError('Three-class native model digest differs')
            import lightgbm as lgb
            self.booster = lgb.Booster(model_file=str(model))
        reference = self.bundle['regression_reference']
        path = root/reference['bundle']
        if not path.resolve().is_relative_to((root/'code/models').resolve()):
            raise ValueError('Regression reference outside model directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
            raise ValueError('Original regression bundle changed')
        self.regression = Predictor(path)
        if self.regression.bundle['code']!=self.bundle['code'] or self.regression.bundle['horizon']!=self.bundle['horizon']:
            raise ValueError('Regression index/horizon mismatch')

    def predict(self, rows):
        if not rows:
            return []
        x = np.asarray([[r[f] for f in self.bundle['features']] for r in rows], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Invalid features')
        spec = self.bundle['classifier']
        if spec['kind'] in ('constant','neutral'):
            p = np.tile(spec['probabilities'],(len(rows),1))
        else:
            if self.booster is not None:
                raw = self.booster.predict(x,num_threads=2)
                if raw.ndim==1:
                    raw=np.column_stack([1-raw,raw])
            else:
                z=(x-np.asarray(spec['mean']))/np.asarray(spec['scale'])
                logits=z@np.asarray(spec['coef']).T+np.asarray(spec['intercept'])
                if len(spec['classes'])==2:
                    up=1/(1+np.exp(-np.clip(logits[:,0],-700,700)))
                    raw=np.column_stack([1-up,up])
                else:
                    exps=np.exp(logits-logits.max(axis=1,keepdims=True))
                    raw=exps/exps.sum(axis=1,keepdims=True)
            p=np.zeros((len(rows),3))
            p[:,np.asarray(spec['classes'],dtype=int)]=raw
        if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
            raise ValueError('Invalid three-class probabilities')
        regression=self.regression.predict(rows)
        return [{'code':self.bundle['code'],'horizon':self.bundle['horizon'],
                 'class_id':int(v.argmax()),'class_label':self.bundle['labels'][int(v.argmax())],
                 'p_big_down':float(v[0]),'p_neutral':float(v[1]),'p_big_up':float(v[2]),
                 'expected_price_return':r['expected_price_return'],'research_only':True}
                for v,r in zip(p,regression)]

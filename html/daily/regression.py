"""Only evaluate the saved return branch; never load classifiers or fit models."""
import hashlib
import json
from pathlib import Path

import numpy as np

from daily.features import FEATURES


class RegressionPredictor:
    def __init__(self, path, expected_sha256=None):
        path=Path(path)
        raw=path.read_bytes()
        if expected_sha256 and hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('回归模型校验失败')
        self.bundle=json.loads(raw)
        if self.bundle.get('schema_version')!=1 or self.bundle['features']!=list(FEATURES):
            raise ValueError('模型版本或特征定义不匹配')
        self.spec=self.bundle['return']
        self.booster=None
        if self.spec['kind']=='lightgbm':
            name=self.spec['file']
            if Path(name).name!=name:
                raise ValueError('模型文件路径无效')
            native=path.parent/name
            if hashlib.sha256(native.read_bytes()).hexdigest()!=self.spec['sha256']:
                raise ValueError('回归权重校验失败')
            import lightgbm as lgb
            self.booster=lgb.Booster(model_file=str(native))
        elif self.spec['kind'] not in ('zero','mean','ridge'):
            raise ValueError('不支持的回归模型')

    def predict(self, row):
        x=np.asarray([[row[f] for f in FEATURES]],dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('特征不完整')
        kind=self.spec['kind']
        if self.booster is not None:
            value=self.booster.predict(x,num_threads=2)[0]
        elif kind in ('zero','mean'):
            value=self.spec['value']
        else:
            z=(x-np.asarray(self.spec['mean']))/np.asarray(self.spec['scale'])
            value=(z@np.asarray(self.spec['coef'])+self.spec['intercept'])[0]
        if not np.isfinite(value):
            raise ValueError('回归输出无效')
        return float(value)

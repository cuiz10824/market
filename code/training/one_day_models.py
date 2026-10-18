"""Small fixed regressors for next-session returns, with no direction classifier."""
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research.one_day import FEATURES


def matrix(rows):
    x=np.asarray([[r[f] for f in FEATURES] for r in rows],dtype=float)
    if not np.isfinite(x).all():raise ValueError('Invalid one-day features')
    return x


def fit(kind,rows,protocol):
    x=matrix(rows)
    y=np.asarray([r['target_return'] for r in rows])
    if kind in ('zero','mean'):
        return {'kind':kind,'value':0. if kind=='zero' else float(y.mean())}
    if kind=='ridge':
        scaler=StandardScaler().fit(x)
        model=Ridge(**protocol['ridge']).fit(scaler.transform(x),y)
        return {'kind':kind,'scaler':scaler,'model':model}
    if kind=='lightgbm':
        model=lgb.LGBMRegressor(objective='regression_l1',random_state=protocol['seed'],**protocol['lightgbm'])
        model.fit(x,y)
        return {'kind':kind,'model':model}
    raise ValueError(kind)


def predict(fitted,rows):
    x=matrix(rows)
    if fitted['kind'] in ('zero','mean'):return np.full(len(rows),fitted['value'])
    if fitted['kind']=='ridge':return fitted['model'].predict(fitted['scaler'].transform(x))
    return fitted['model'].booster_.predict(x,num_threads=2)

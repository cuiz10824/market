"""Fixed-capacity candidates with native artifact export."""
import hashlib

import lightgbm as lgb
from catboost import CatBoostRegressor


def fit(kind,rows,features,p):
    x=rows[list(features)].to_numpy();y=rows.target_return.to_numpy()
    if kind.startswith('lightgbm'):
        model=lgb.LGBMRegressor(objective='regression_l1' if kind.endswith('mae') else 'regression',
                               random_state=p['seed'],**p['lightgbm'])
    else:
        model=CatBoostRegressor(loss_function='MAE' if kind.endswith('mae') else 'RMSE',
                                random_seed=p['seed'],**p['catboost'])
    model.fit(x,y)
    return model


def predict(model,kind,rows,features):
    x=rows[list(features)].to_numpy()
    return model.booster_.predict(x,num_threads=2) if kind.startswith('lightgbm') else model.predict(x,thread_count=2)


def export(model,kind,folder):
    path=folder/('return.lgb.txt' if kind.startswith('lightgbm') else 'return.cbm')
    if kind.startswith('lightgbm'):model.booster_.save_model(str(path))
    else:model.save_model(str(path))
    return {'kind':kind,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

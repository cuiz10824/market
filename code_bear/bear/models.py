"""Small baselines with JSON/native-LightGBM weights, no pickle."""
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit(kind,x,y,protocol):
    counts=np.bincount(y,minlength=3)
    if (counts==0).any():raise ValueError('Each training split must contain all3 regimes')
    if kind in ('majority','persistence','ma_rule'):
        return dict(kind=kind,priors=(counts/counts.sum()).tolist())
    if kind=='logistic':
        scaler=StandardScaler().fit(x)
        model=LogisticRegression(C=1.,max_iter=2000,solver='lbfgs').fit(scaler.transform(x),y)
        if (model.n_iter_>=2000).any():raise ValueError('Logistic regression failed to converge')
        return dict(kind=kind,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                    coef=model.coef_.tolist(),intercept=model.intercept_.tolist())
    if kind=='lightgbm':
        model=lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=100,num_leaves=7,
             max_depth=3,min_child_samples=100,learning_rate=.03,reg_lambda=10.,
             random_state=protocol['seed'],n_jobs=protocol['threads'],verbosity=-1,
             deterministic=True,force_col_wise=True)
        model.fit(x,y)
        return dict(kind=kind,booster=model.booster_)
    raise ValueError('Unknown model')


def predict(model,x,persistence,ma):
    kind=model['kind']
    if kind=='majority':out=np.tile(model['priors'],(len(x),1))
    elif kind in ('persistence','ma_rule'):
        labels=np.asarray(persistence if kind=='persistence' else ma,dtype=int)
        if (labels<0).any():raise ValueError('Unconfirmed persistence input')
        out=np.eye(3)[labels]
    elif kind=='logistic':
        z=(np.asarray(x)-np.asarray(model['mean']))/np.asarray(model['scale'])
        logits=z@np.asarray(model['coef']).T+np.asarray(model['intercept'])
        out=np.exp(logits-logits.max(axis=1,keepdims=True));out/=out.sum(axis=1,keepdims=True)
    else:out=model['booster'].predict(x)
    if out.shape!=(len(x),3) or not np.isfinite(out).all():raise ValueError('Invalid class scores')
    np.testing.assert_allclose(out.sum(axis=1),1.,atol=1e-9)
    return out


def save(model,folder,metadata):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    bundle={k:v for k,v in model.items() if k!='booster'}
    if model['kind']=='lightgbm':
        path=folder/'model.lgb.txt';model['booster'].save_model(str(path))
        bundle['native']=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    bundle.update(schema_version=1,task='sh000001_forward_regime',metadata=metadata)
    path=folder/'model.json';path.write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


def load(path):
    path=Path(path);bundle=json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('task')!='sh000001_forward_regime' or bundle.get('schema_version')!=1:raise ValueError('Wrong regime model')
    if 'native' in bundle:
        spec=bundle['native']
        if Path(spec['file']).name!=spec['file']:raise ValueError('Invalid native path')
        native=path.parent/spec['file']
        if hashlib.sha256(native.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Native weight mismatch')
        bundle['booster']=lgb.Booster(model_file=str(native))
    return bundle

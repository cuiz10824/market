"""Inference only: saved selected models against the exact versioned data snapshot."""
import json

import numpy as np
import pandas as pd

from train import ROOT,read,sha,write
from bear.dataset import features,targets,ma_state,NAMES
from bear.models import load,predict


def run():
    pointer=read(ROOT/'models/latest.json');s=read(ROOT/pointer['report'])
    data=read(ROOT/'data/latest.json');manifest=read(ROOT/data['manifest'])
    if sha(ROOT/data['csv'])!=manifest['csv_sha256']:raise ValueError('Dataset checksum differs')
    frame=pd.read_csv(ROOT/data['csv']);original=pd.read_csv(ROOT/s['dataset']['csv'])
    overlap=original.merge(frame,on='date',suffixes=('_train','_now'),validate='one_to_one')
    if len(overlap)!=len(original):raise ValueError('New dataset is missing training history')
    for field in ('open','high','low','close'):
        np.testing.assert_allclose(overlap[field+'_train'],overlap[field+'_now'],atol=.020001,rtol=0)
    np.testing.assert_allclose(overlap.volume_train,overlap.volume_now,atol=1.,rtol=.00001)
    for name,digest in s['source_sha256'].items():
        if name.replace('\\','/').startswith('bear/') and sha(ROOT/name)!=digest:raise ValueError('Model feature source changed')
    x=features(frame);ma=ma_state(frame);items=[]
    checks=read((ROOT/pointer['report']).parent/'checksums.json')
    index=read(ROOT/pointer['index'])
    for h in s['protocol']['horizons']:
        kind=s['choices'][str(h)]['candidate']
        entry=next(e for e in index['models'] if e['role']=='latest' and e['horizon']==h and e['candidate']==kind)
        if sha(ROOT/entry['bundle'])!=checks[entry['bundle']]:raise ValueError('Selected model bundle changed')
        model=load(ROOT/entry['bundle'])
        if model['metadata']['features']!=list(x.columns):raise ValueError('Feature order changed')
        if model['metadata']['train_label_end']>frame.date.iloc[-1]:raise ValueError('Prediction precedes model labels')
        target=targets(frame,h,s['protocol']['thresholds'][str(h)],s['protocol']['path_occupancy'])
        past=target.label.shift(h).fillna(-1).to_numpy(dtype=int)
        value=predict(model,x.iloc[[-1]],past[-1:],ma[-1:])[0]
        items.append(dict(horizon=h,candidate=kind,as_of=frame.date.iloc[-1],label=NAMES[int(value.argmax())],
                          scores=value.tolist(),score_type='deterministic_rule' if kind in ('majority','persistence','ma_rule') else 'uncalibrated_model_scores'))
    output=dict(run_id=s['run_id'],dataset=data,as_of=manifest['last_date'],predictions=items,research_only=True)
    write(ROOT/'output/current_prediction.json',output)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':run()

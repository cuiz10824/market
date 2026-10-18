"""No fitting: replay every safetensors artifact, labels, cutoffs and ensemble choices."""
import csv
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from research.sequence_data import SequenceData,CHANNELS
from training.neural_engine import load,predict,scaled_score
from training.train_enhanced import read,sha,write,metrics,old_reports
from training.train_neural import group_scores,legacy_validation
from training.provenance import historical_source


def assert_metrics(actual,expected):
    for key in ('mae_pp','bias_pp','error_std_pp','rmse_pp'):
        assert abs(actual[key]-expected[key])<1e-9,(key,actual,expected)
    assert actual['n']==expected['n']


def run():
    pointer=read(ROOT/'code/models/latest_neural.json');s=read(ROOT/pointer['report']);p=s['protocol']
    index=read(ROOT/pointer['index']);verification=read(ROOT/pointer['verification'])
    assert pointer['research_only'] and not pointer['auto_deploy'] and not index['auto_deploy']
    for group in ('all_saved_file_sha256','original_artifact_sha256'):
        for path,digest in verification[group].items():assert sha(ROOT/path)==digest,path
    for path,digest in s['source_sha256'].items():historical_source(ROOT,path,digest)
    report=(ROOT/pointer['report']).parent
    assert sha(report/'frozen_protocol.json')==s['protocol_sha256']
    assert sha(ROOT/'data/enhanced/market_ohlcv.sqlite3')==s['database_sha256']
    torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    data=SequenceData(p);old,original,_=old_reports();cache={};native_rows=0
    for e in index['models']:
        net,bundle=load(ROOT/e['bundle']);m=bundle['metadata'];length=e['lookback'];role=e['role']
        assert bundle['codes']==p['codes'] and bundle['horizons']==p['horizons']
        assert bundle['channels']==list(CHANNELS)
        assert bundle['sequence_length']==data.x[length].shape[1]
        assert bundle['target_scales']==data.target_scales.tolist()
        assert m['parameter_count']==sum(t.numel() for t in net.parameters())
        train,tm=data.split(e['start'],train=True,stride=p['train_stride'])
        assert m['training_rows']==len(train)
        assert m['fit_origin_start']==min(data.dates[train]) and m['fit_origin_end']==max(data.dates[train])
        for j,h in enumerate(data.horizons):
            assert m['label_counts'][str(h)]==int(tm[:,j].sum())
            assert m['fit_label_ends'][str(h)]==max(data.calendar[data.label_indices[train,j][tm[:,j]]])
            if e['start'] is not None:assert m['fit_label_ends'][str(h)]<e['start']
        log=read(ROOT/e['log']);assert log['selected_epoch']==m['selected_epoch']
        if role in ('screen','replication'):
            rows,mask=data.split(e['start'],e['end']);artifact=e['validation']
            eligible=[v for v in log['history'] if v['epoch']>=p['minimum_epochs']]
            winner=min(eligible,key=lambda v:v['validation_scaled_mae'])
            selected=next(v for v in eligible if v['epoch']==m['selected_epoch'])
            assert selected['validation_scaled_mae']<=winner['validation_scaled_mae']+1.01e-8
        elif role=='holdout':
            rows,mask=data.split(e['start'])
            targets=['000510'] if e['start']==p['a500_holdout_start'] else [c for c in data.codes if c!='000510']
            keep=np.isin(data.ids[rows],[data.codes.index(c) for c in targets]);rows=rows[keep];mask=mask[keep]
            artifact=e['test'];assert m['selected_epoch']==s['refit_epochs'][e['architecture']]
        else:
            rows=np.array([np.where((data.ids==ci)&(data.ends==len(data.calendar)-1))[0][0] for ci in range(len(data.codes))])
            mask=None;artifact=e['latest'];assert m['selected_epoch']==s['refit_epochs'][e['architecture']]
        with np.load(ROOT/artifact,allow_pickle=False) as a:
            assert np.array_equal(rows,a['rows'])
            if mask is not None:assert np.array_equal(mask,a['mask'])
            value=predict(net,data,length,rows)
            np.testing.assert_allclose(value,a['prediction'],atol=1e-10,rtol=1e-8)
        if role in ('screen','replication'):
            assert abs(e['score']-scaled_score(data,rows,mask,value))<1e-10
            ms=group_scores(data,rows,mask,value)
            for key,score in ms.items():assert_metrics(score,e['task_scores'][key])
        cache[(role,e['architecture'],length,e['start'],e['seed'])]=(rows,mask,value)
        native_rows+=len(rows)
    # Validate screening and ensemble development metrics directly from native outputs.
    for architecture in p['architectures']:
        scores={}
        for length in p['lookbacks']:
            entries=[e for e in index['models'] if e['role']=='screen' and e['architecture']==architecture and e['lookback']==length]
            assert len(entries)==3
            scores[length]=float(np.mean([e['score'] for e in entries]))
            assert abs(scores[length]-s['screening'][architecture][str(length)])<1e-10
        chosen=min(p['lookbacks'],key=lambda n:scores[n]);assert chosen==s['chosen_lookbacks'][architecture]
        selected_epochs=[]
        for fold in s['development'][architecture]:
            start=fold['start'];length=chosen
            rows,mask,one=cache[('screen',architecture,length,start,p['screening_seed'])]
            rr,mm,two=cache[('replication',architecture,length,start,p['replication_seed'])]
            assert np.array_equal(rows,rr) and np.array_equal(mask,mm)
            computed=group_scores(data,rows,mask,(one+two)/2)
            zero=group_scores(data,rows,mask,np.zeros_like(one))
            train,tm=data.split(start,train=True,stride=p['train_stride']);baseline=np.zeros_like(one)
            for ci in range(len(data.codes)):
                for j in range(len(data.horizons)):
                    good=(data.ids[train]==ci)&tm[:,j]
                    baseline[data.ids[rows]==ci,j]=float(np.mean(data.y[train[good],j]))
            means=group_scores(data,rows,mask,baseline)
            for key in computed:
                assert_metrics(computed[key],fold['ensemble'][key]);assert_metrics(zero[key],fold['zero'][key]);assert_metrics(means[key],fold['mean'][key])
            selected_epochs.extend([e['selected_epoch'] for e in index['models'] if e['role'] in ('screen','replication') and e['architecture']==architecture and e['lookback']==chosen and e['start']==start])
        assert int(np.floor(np.median(selected_epochs)+.5))==s['refit_epochs'][architecture]
    with (report/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    positions={key:{int(row):i for i,row in enumerate(value[0])} for key,value in cache.items()}
    origins={(data.codes[data.ids[i]],str(data.dates[i])):i for i in range(len(data.ids))}
    groups={};duplicates=set()
    for r in records:
        code=r['code'];h=int(r['horizon']);a=r['candidate'];length=int(r['lookback']);j=data.horizons.index(h)
        identity=(code,h,r['date'],a);assert identity not in duplicates;duplicates.add(identity)
        ix=origins[(code,r['date'])]
        assert data.known[ix,j]
        assert r['label_end_date']==str(data.calendar[data.label_indices[ix,j]])
        assert abs(float(r['actual_return'])-data.y[ix,j])<1e-12
        assert abs(float(r['actual_return'])-float(original[(code,h,r['date'])]['actual_return']))<1e-12
        assert float(r['legacy_return'])==float(original[(code,h,r['date'])]['expected_return'])
        start=p['a500_holdout_start'] if code=='000510' else p['holdout_starts'][0]
        values=[]
        for seed in (p['screening_seed'],p['replication_seed']):
            key=('holdout',a,length,start,seed);values.append(float(cache[key][2][positions[key][ix],j]))
        assert abs(values[0]-float(r['seed1_return']))<1e-12 and abs(values[1]-float(r['seed2_return']))<1e-12
        assert abs(float(np.mean(values))-float(r['expected_return']))<1e-12
        groups.setdefault((code,h,a),[]).append(r)
    for t in s['tasks']:
        key=f'{t["code"]}:{t["horizon"]}';means=t['validation_mean_mae_pp']
        for a in p['architectures']:
            assert abs(means[a]-np.mean([f['ensemble'][key]['mae_pp'] for f in s['development'][a]]))<1e-10
            rows=groups[(t['code'],t['horizon'],a)]
            assert_metrics(metrics([float(r['actual_return']) for r in rows],[float(r['expected_return']) for r in rows]),t['test_candidates'][a])
            length=s['chosen_lookbacks'][a];j=data.horizons.index(t['horizon']);ci=data.codes.index(t['code'])
            values=[cache[('latest',a,length,None,seed)][2][ci,j] for seed in (p['screening_seed'],p['replication_seed'])]
            assert abs(float(np.mean(values))-t['latest_candidate_returns'][a])<1e-12
        legacy=float(np.mean([legacy_validation(old,t['code'],t['horizon'],f[0]) for f in p['development_windows']]))
        assert abs(legacy-means['legacy'])<1e-10
        for baseline in ('zero','mean'):
            expected=np.mean([f[baseline][key]['mae_pp'] for f in s['development'][p['architectures'][0]]])
            assert abs(expected-means[baseline])<1e-10
        reference=groups[(t['code'],t['horizon'],p['architectures'][0])]
        actual=[float(r['actual_return']) for r in reference]
        assert_metrics(metrics(actual,[float(r['legacy_return']) for r in reference]),t['test_legacy'])
        assert_metrics(metrics(actual,[0.]*len(reference)),t['test_zero'])
        winner=min(p['architectures'],key=lambda a:means[a]);gain=1-means[winner]/min(means[k] for k in ('zero','mean','legacy'))
        assert t['choice']==(winner if gain>=p['min_relative_validation_gain'] else 'legacy')
    assert len(records)==verification['candidate_predictions']
    verification.update(status='independent_replay_passed',independent_replay_passed=True,native_prediction_rows_replayed=native_rows,
                        ensemble_records_replayed=len(records),fit_called_during_verification=False)
    write(ROOT/pointer['verification'],verification)
    print(f'PASS: {len(index["models"])} native bundles; {len(records)} ensemble records; all temporal masks, labels, epochs, choices and original artifacts verified.')


if __name__=='__main__':run()

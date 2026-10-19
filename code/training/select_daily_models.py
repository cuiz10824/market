"""User-authorized daily selection: lowest validation MAE among fitted regressors.
Zero and historical-mean baselines are excluded; test scores do not choose models.
"""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'html')]
from daily.selection import load_selection


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def build(root=ROOT):
    items=load_selection(root);codes=[x['code'] for x in items];horizons=[1,5,21,63,126]
    pools={};sources={};digests={};database=set()
    for pointer_name,profile in [('latest_enhanced','rich'),('latest_five_year','cycle'),('latest_neural','neural')]:
        pointer=read(root/'code/models'/f'{pointer_name}.json')
        summary=read(root/pointer['report']);index=read(root/pointer['index']);verification=read(root/pointer['verification'])
        if not verification.get('independent_replay_passed'):raise ValueError('Candidate run lacks independent replay: '+pointer_name)
        sources[profile]={'run_id':pointer['run_id'],'report':pointer['report'],'report_sha256':sha(root/pointer['report']),
                          'index':pointer['index'],'index_sha256':sha(root/pointer['index'])}
        digests.update(verification['all_saved_file_sha256']);database.add(summary['database_sha256'])
        for task in summary['tasks']:
            code=task['code'];h=task['horizon']
            if code not in codes:continue
            for candidate in summary['protocol'].get('candidates',summary['protocol'].get('architectures',[])):
                if profile=='neural':
                    bundles=[e['bundle'] for e in index['models'] if e['role']=='latest' and e['architecture']==candidate]
                    length=summary['chosen_lookbacks'][candidate]
                    counts=[f['ensemble'][f'{code}:{h}']['n'] for f in summary['development'][candidate]]
                    effective_profile='neural';family={'dlinear':'DLinear','nlinear':'NLinear','tcn':'TCN','patchtst':'PatchTST'}[candidate]
                else:
                    bundles=[e['bundle'] for e in index['models'] if e['role']=='latest' and e['candidate']==candidate and e['horizon']==h]
                    length=1260 if candidate.endswith('_5y') else 252
                    counts=[f['results'][code][candidate]['n'] for f in summary['development_folds'][str(h)]]
                    effective_profile='cycle' if length==1260 else 'rich';family='CatBoost' if candidate.startswith('catboost') else 'LightGBM'
                if len(bundles)!=(2 if profile=='neural' else 1):raise ValueError('Incomplete latest model ensemble')
                entry=dict(id=pointer['run_id']+'/'+candidate,source=profile,profile=effective_profile,candidate=candidate,
                           run_id=pointer['run_id'],code=code,horizon=h,lookback=length,bundles=sorted(bundles),
                           label=family+' · '+str({252:1,756:3,1260:5}[length])+'年',
                           validation_mae_pp=task['validation_mean_mae_pp'][candidate],validation_counts=counts,
                           zero_validation_mae_pp=task['validation_mean_mae_pp']['zero'],test_metrics=task['test_candidates'][candidate],
                           previous_test_metrics=task['test_legacy'])
                pools.setdefault((code,h),[]).append(entry)
    # Include saved legacy learned regressors, but not constant zero/mean bundles.
    reference=read(root/sources['neural']['report']);reference_tasks={(t['code'],t['horizon']):t for t in reference['tasks']}
    for name in ('latest','latest_one_day'):
        pointer=read(root/'code/models'/f'{name}.json');index=read(root/pointer['index'])
        verification=read(root/pointer['verification']);digests.update(verification['all_saved_file_sha256'])
        for e in index['models']:
            code=e['code'];h=e['horizon']
            if code not in codes:continue
            bundle=read(root/e['latest']);kind=bundle['return']['kind']
            if kind not in ('lightgbm','ridge'):continue
            task=reference_tasks[(code,h)];base=pools[(code,h)][0]
            pools[(code,h)].append(dict(id=pointer['run_id']+'/'+kind,source='legacy',profile='legacy',candidate=kind,run_id=pointer['run_id'],
                 code=code,horizon=h,lookback=252,bundles=[e['latest']],label=('LightGBM' if kind=='lightgbm' else '岭回归')+' · 1年',
                 validation_mae_pp=task['validation_mean_mae_pp']['legacy'],validation_counts=base['validation_counts'],
                 zero_validation_mae_pp=task['validation_mean_mae_pp']['zero'],test_metrics=task['test_legacy'],previous_test_metrics=task['test_legacy']))
    choices=[]
    for code in codes:
        for h in horizons:
            candidates=pools[(code,h)]
            if len({tuple(c['validation_counts']) for c in candidates})!=1:raise ValueError('Validation observation counts differ')
            chosen=min(candidates,key=lambda c:(c['validation_mae_pp'],c['id']))
            choices.append(dict(chosen,candidate_scores=[{'id':c['id'],'validation_mae_pp':c['validation_mae_pp']} for c in candidates]))
    if len(database)!=1:raise ValueError('Candidates use different frozen OHLCV sources')
    runtime_paths=['html/daily/neural_architecture.py','html/daily/neural_features.py','html/daily/neural_predictor.py',
                   'html/daily/rich_features.py','html/daily/cycle_features.py']
    selected_paths={path for choice in choices for path in choice['bundles']}
    for path in selected_paths:
        if sha(root/path)!=digests[path]:raise ValueError('Selected bundle fingerprint differs')
    manifest=dict(schema_version=1,policy='lowest_rolling_validation_mae_among_fitted_regressors',
                  exclude_constant_baselines=True,require_improvement_over_zero=False,
                  generated_at=dt.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),codes=codes,horizons=horizons,
                  database_sha256=next(iter(database)),sources=sources,tasks=choices,
                  bundle_sha256={path:digests[path] for path in sorted(selected_paths)},
                  runtime_sha256={path:sha(root/path) for path in runtime_paths})
    stable={k:v for k,v in manifest.items() if k!='generated_at'}
    manifest['selection_id']=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()[:16]
    write(root/'html/daily/model_selection.json',manifest)
    folder=root/'data/reports/deployment'/manifest['generated_at'][:10];folder.mkdir(parents=True,exist_ok=True)
    flat=[]
    for c in choices:
        flat.append(dict(code=c['code'],horizon=c['horizon'],model=c['label'],candidate_id=c['id'],validation_mae_pp=c['validation_mae_pp'],
                         beats_zero_validation=c['validation_mae_pp']<c['zero_validation_mae_pp'],
                         test_mae_pp=c['test_metrics']['mae_pp'],previous_test_mae_pp=c['previous_test_metrics']['mae_pp']))
    with (folder/'selection.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# 每日网页：最优已训练回归候选','',f'选择版本 `{manifest["selection_id"]}`。按用户要求改用40组最优已训练回归输出，不再要求优于固定零变化基准。','',
           '每个行业期限仅按2018—2023三个滚动验证窗的平均MAE选型；树模型与神经网络比较相同验证观测数量。测试误差仅报告，不参与选择。排除固定零和历史均值模型；保留原有已学习的LightGBM/岭回归候选。','',
           '这会采用部分尚未胜过零基准的模型，不等于准确率已有改善。输出直接取保存权重，不放大、不添加人为涨跌、不以最乐观预测选模型。最新量价不可用时使用最后可用收盘日期并逐格标明；没有合格输入则显示不可用，不能补零。','',
           '|期限（交易日）|原模型测试MAE|新选择测试MAE|','|---|---:|---:|']
    for h in horizons:
        group=[c for c in choices if c['horizon']==h]
        lines.append(f'|{h}|{sum(c["previous_test_metrics"]["mae_pp"] for c in group)/8:.3f}|{sum(c["test_metrics"]["mae_pp"] for c in group)/8:.3f}|')
    lines+=['','单位为百分点，8指数等权；测试价格区间已被观察，是探索性历史复测。','',
            '- [40组选择明细](selection.csv)。运行配置为html/daily/model_selection.json。',
            '- 复现：`.venv/bin/python code/training/select_daily_models.py`；没有训练或改写原模型。',
            '- `.venv/bin/python html/predict.py`生成单文件快照；`--refresh`额外联网更新已收盘行情。','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('SELECTION',manifest['selection_id'])
    from collections import Counter
    print('Selected:',dict(Counter(c['source']+':'+c['candidate'] for c in choices)))
    return manifest


if __name__=='__main__':build()

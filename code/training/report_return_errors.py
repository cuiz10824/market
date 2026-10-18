"""Recompute return-error moments from saved holdout predictions, without fitting."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'html'))
from daily.selection import load_selection

HORIZONS=(1,5,21,63,126)
NAMES={1:'下一交易日',5:'一周',21:'一月',63:'三月',126:'六月'}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def error_statistics(predicted,actual):
    if not predicted or len(predicted)!=len(actual):raise ValueError('Missing or mismatched pairs')
    if not all(math.isfinite(v) for v in predicted+actual):raise ValueError('Nonfinite return')
    errors=[(p-y)*100 for p,y in zip(predicted,actual)]
    absolute=[abs(e) for e in errors]
    bias=statistics.mean(errors)
    mse=statistics.mean([e*e for e in errors])
    variance=statistics.pvariance(errors)
    return {'n':len(errors),'prediction_mean_pp':statistics.mean(predicted)*100,
            'actual_mean_pp':statistics.mean(actual)*100,'mae_pp':statistics.mean(absolute),
            'bias_pp':bias,'error_variance_pp2':variance,'error_std_pp':math.sqrt(variance),
            'absolute_error_variance_pp2':statistics.pvariance(absolute),
            'mse_pp2':mse,'rmse_pp':math.sqrt(mse)}


def task_statistics(rows):
    predicted=[float(r['expected_return']) for r in rows]
    actual=[float(r['actual_return']) for r in rows]
    selected=error_statistics(predicted,actual)
    zero=error_statistics([0.]*len(rows),actual)
    means=error_statistics([float(r['historical_mean_return']) for r in rows],actual)
    last_end='';nonoverlap=0
    for row in rows:
        if row['date']>=last_end:nonoverlap+=1;last_end=row['label_end_date']
    result={'selected':selected,'zero':zero,'historical_mean':means,
            'mae_improvement_vs_zero_pp':zero['mae_pp']-selected['mae_pp'],
            'mae_improvement_vs_zero_fraction':(zero['mae_pp']-selected['mae_pp'])/zero['mae_pp'] if zero['mae_pp'] else None,
            'variance_reduction_vs_zero_pp2':zero['error_variance_pp2']-selected['error_variance_pp2'],
            'zero_predictions':sum(v==0 for v in predicted),'nonoverlapping_intervals':nonoverlap}
    return result


def csvwrite(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    items=load_selection(ROOT);codes=[i['code'] for i in items]
    groups={};metadata={};source_sha={};source_runs=[]
    profile=ROOT/'data/active_indices.json'
    source_sha[str(profile.relative_to(ROOT))]=digest(profile)
    for pointer_name in ('latest.json','latest_one_day.json'):
        pointer_path=ROOT/'code/models'/pointer_name
        pointer=read(pointer_path);folder=ROOT/'data/reports/training'/pointer['run_id']
        summary=read(folder/'summary.json');source_runs.append(pointer['run_id'])
        for path in (pointer_path,folder/'summary.json',folder/'test_predictions.csv'):
            source_sha[str(path.relative_to(ROOT))]=digest(path)
        one_day=pointer_name=='latest_one_day.json'
        for task in summary['tasks']:
            if task['code'] in codes:
                metadata[(task['code'],task['horizon'])]=(task,one_day)
        with (folder/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                code=row['code'];h=int(row['horizon'])
                if code not in codes:continue
                if (h==1)!=one_day:raise ValueError('Horizon appears in wrong run')
                if row['date']>=row['label_end_date']:raise ValueError('Non-forward target')
                groups.setdefault((code,h),[]).append(row)
    if set(groups)!={(code,h) for code in codes for h in HORIZONS}:raise ValueError('Incomplete/duplicate model scope')
    tasks=[];yearly=[];flat=[]
    for item in items:
        for h in HORIZONS:
            key=(item['code'],h);rows=groups[key]
            if [r['date'] for r in rows]!=sorted({r['date'] for r in rows}):raise ValueError('Repeated or unordered predictions')
            recorded,one_day=metadata[key]
            model=recorded['choice'] if one_day else recorded['choices']['return']
            stats=task_statistics(rows)
            expected=recorded['test']['selected' if one_day else 'selected_return']
            for name,field in [('mae','mae_pp'),('rmse','rmse_pp'),('bias','bias_pp')]:
                if abs(expected[name]*100-stats['selected'][field])>1e-10:
                    raise ValueError('Recalculation differs from original report')
            n=recorded['test']['n'] if one_day else recorded['test']['selected_direction']['n']
            if n!=len(rows):raise ValueError('Original row count differs')
            task={'code':item['code'],'name':item['name'],'display_name':item['display_name'],
                  'horizon':h,'model':model,'test_start':rows[0]['date'],'test_origin_end':rows[-1]['date'],
                  'last_label_end':max(r['label_end_date'] for r in rows),**stats}
            tasks.append(task)
            flat.append({'code':item['code'],'name':item['name'],'display_name':item['display_name'],
                         'horizon':h,'model':model,'test_start':task['test_start'],'test_origin_end':task['test_origin_end'],
                         'last_label_end':task['last_label_end'],**stats['selected'],
                         'zero_mae_pp':stats['zero']['mae_pp'],'zero_bias_pp':stats['zero']['bias_pp'],
                         'zero_error_variance_pp2':stats['zero']['error_variance_pp2'],
                         'mae_improvement_vs_zero_pp':stats['mae_improvement_vs_zero_pp'],
                         'variance_reduction_vs_zero_pp2':stats['variance_reduction_vs_zero_pp2'],
                         'zero_predictions':stats['zero_predictions'],'nonoverlapping_intervals':stats['nonoverlapping_intervals']})
            for year in sorted({r['date'][:4] for r in rows}):
                subset=[r for r in rows if r['date'].startswith(year)]
                metrics=task_statistics(subset)
                yearly.append({'code':item['code'],'display_name':item['display_name'],'horizon':h,
                               'origin_year':year,'model':model,**metrics['selected'],
                               'zero_mae_pp':metrics['zero']['mae_pp'],
                               'zero_error_variance_pp2':metrics['zero']['error_variance_pp2'],
                               'mae_improvement_vs_zero_pp':metrics['mae_improvement_vs_zero_pp']})
    aggregate=[]
    for h in HORIZONS:
        selected=[t for t in tasks if t['horizon']==h]
        pooled=[r for code in codes for r in groups[(code,h)]]
        fields=('mae_pp','bias_pp','error_variance_pp2','error_std_pp','absolute_error_variance_pp2','rmse_pp')
        pooled_stats=task_statistics(sorted(pooled,key=lambda r:r['date']))
        pooled_stats.pop('nonoverlapping_intervals')  # Intervals are defined per index, not across indices.
        aggregate.append({'horizon':h,'tasks':len(selected),'test_rows':len(pooled),
                          'index_equal_weight':{k:statistics.mean([t['selected'][k] for t in selected]) for k in fields},
                          'zero_index_equal_weight':{k:statistics.mean([t['zero'][k] for t in selected]) for k in fields},
                          'mae_better_than_zero':sum(t['mae_improvement_vs_zero_pp']>1e-12 for t in selected),
                          'pooled':pooled_stats})
    date=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    folder=ROOT/'data/reports/performance'/('eight_indices_'+date)
    folder.mkdir(parents=True,exist_ok=True)
    summary={'as_of':date,'source_runs':source_runs,'source_sha256':source_sha,
             'method':{'error':'100*(predicted_decimal_return-actual_decimal_return)',
                       'mean':'signed error mean (bias); separately report MAE',
                       'variance':'population variance of signed errors, ddof=0, unit percentage-point squared',
                       'absolute_error_variance':'population variance of absolute errors, ddof=0',
                       'aggregate':'equal-index means; pooled moments separately labeled',
                       'model_role':'saved holdout predictions; not latest-refit independent evaluation'},
             'tasks':tasks,'aggregate_by_horizon':aggregate}
    write(folder/'summary.json',summary)
    csvwrite(folder/'by_index_horizon.csv',flat)
    csvwrite(folder/'by_year_index_horizon.csv',yearly)
    lines=['# 预测涨跌幅与真实涨跌幅的差距：均值与方差','',
           '范围为当前8个指数×下一交易日、一周、一月、三月、六月，共40组。使用已经保存的历史测试预测重新统计，未重新训练、未更换模型。CatBoost、Kronos等候选尚未在本项目训练，本表不包含其成绩。','',
           '## 统计口径','',
           '- 误差 e = 预测涨跌幅 − 真实涨跌幅，先换算为百分点。正误差表示预测偏高，负误差表示预测偏低。',
           '- 平均偏差 = mean(e)；平均绝对误差 MAE = mean(|e|)。正负误差会抵消，平均偏差接近0不能单独说明准确。',
           '- 误差方差 = mean((e − mean(e))²)，采用总体方差ddof=0，单位为百分点²。另保存绝对误差方差、标准差、RMSE。',
           '- MSE = 误差方差 + 平均偏差²；MSE = 绝对误差方差 + MAE²。小方差也可能伴随很大的系统偏差。',
           '- 以下汇总对8个指数等权平均；“平均方差”是各指数方差的均值，不是把全部指数混在一起计算的方差。混合样本指标单列在JSON的pooled中。',
           '- 一般测试从2024-01-02起，中证A500从2025-01-02起。标签截至2026-10-16，各期限可用预测起点不同。测试对应holdout固定参数，不能当作网页latest重训参数的新独立测试成绩。','',
           '## 按期限汇总','',
           '|期限|平均绝对误差MAE|平均偏差|平均误差方差|平均标准差|零预测MAE|MAE优于零预测|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in aggregate:
        m=row['index_equal_weight'];z=row['zero_index_equal_weight']
        lines.append(f'|{NAMES[row["horizon"]]}|{m["mae_pp"]:.3f}|{m["bias_pp"]:+.3f}|{m["error_variance_pp2"]:.3f}|{m["error_std_pp"]:.3f}|{z["mae_pp"]:.3f}|{row["mae_better_than_zero"]}/8|')
    lines+=['','MAE、平均偏差、标准差为百分点；方差为百分点²。不同量纲的数字不能直接比较。','',
            '## 各指数：平均绝对误差 / 误差方差','',
            '每格前者为MAE（百分点），后者为带符号误差的总体方差（百分点²）。','',
            '|方向 / 指数|下一交易日|一周|一月|三月|六月|','|---|---:|---:|---:|---:|---:|']
    lookup={(t['code'],t['horizon']):t for t in tasks}
    for item in items:
        cells=[]
        for h in HORIZONS:
            m=lookup[(item['code'],h)]['selected'];cells.append(f'{m["mae_pp"]:.3f} / {m["error_variance_pp2"]:.3f}')
        lines.append('|'+item['display_name']+' · '+item['name']+'|'+'|'.join(cells)+'|')
    lines+=['','## 各任务的均值、方差与基准','',
            '|方向|期限|模型|样本数|平均预测|平均真实|平均偏差|MAE|误差方差|绝对误差方差|零预测MAE|',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in tasks:
        m=t['selected']
        lines.append(f'|{t["display_name"]}|{NAMES[t["horizon"]]}|{t["model"]}|{m["n"]}|{m["prediction_mean_pp"]:+.3f}|{m["actual_mean_pp"]:+.3f}|{m["bias_pp"]:+.3f}|{m["mae_pp"]:.3f}|{m["error_variance_pp2"]:.3f}|{m["absolute_error_variance_pp2"]:.3f}|{t["zero"]["mae_pp"]:.3f}|')
    lines+=['','## 如何判断效果','',
            f'- 当前40组中，{sum(t["model"]=="zero" for t in tasks)}组选择固定零收益；其MAE和误差方差与零预测基准完全相同。',
            '- 历史平均收益是常数，相比零预测只平移误差，误差方差理论上相同，MAE和偏差可能改变。因此仅看方差会漏掉系统性偏差。',
            '- 小误差不证明模型有信息优势；需要同时比较零预测、分年稳定性和偏差。平均值也不是每次误差的上限。',
            '- 多日收益窗口重叠，指数之间也相关；本表是已观察测试样本的描述统计，不提供独立样本显著性结论。半年非重叠区间很少，数量在CSV中单列。',
            '- 预测的是指数价格收益，不含基金分红、费用和跟踪误差。','',
            '## 文件与复现','',
            '- [40组完整明细](by_index_horizon.csv)：含误差均值、MAE、两种方差、标准差、RMSE与零预测对照，UTF-8 BOM可用Excel打开。',
            '- [逐年逐指数逐期限明细](by_year_index_horizon.csv)：按预测起始年份统计。',
            '- [完整JSON及来源指纹](summary.json)：包含逐任务、等权汇总及混合样本统计。',
            '- 仓库根目录运行 `python code/training/report_return_errors.py`，只读取既有预测，不训练模型。','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    for path,value in source_sha.items():
        if digest(ROOT/path)!=value:raise ValueError('Source changed during reporting')
    write(folder/'verification.json',{'status':'passed','tasks':len(tasks),
          'source_prediction_rows_used':sum(len(v) for v in groups.values()),'yearly_groups':len(yearly),
          'original_mae_rmse_bias_reproduced':True,'model_training_called':False,'sources_unchanged':True})
    print(folder/'README.md')
    for row in aggregate:print(NAMES[row['horizon']],row['index_equal_weight'])


if __name__=='__main__':run()

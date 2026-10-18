"""Report fixed five-year experiment and date-matched one-year controls."""
import csv
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from training.train_enhanced import read,write,metrics
from training.report_enhanced import csvwrite,TITLES
from daily.selection import load_selection
from research.cycle_dataset import load_bars,samples

LABELS={'legacy':'当前模型','lightgbm_mae_1y':'LightGBM一年对照','catboost_mae_1y':'CatBoost一年对照',
        'lightgbm_mae_5y':'LightGBM五年','catboost_mae_5y':'CatBoost五年'}


def report():
    pointer=read(ROOT/'code/models/latest_five_year.json');s=read(ROOT/pointer['report'])
    folder=(ROOT/pointer['report']).parent;p=s['protocol'];tasks=s['tasks']
    items={i['code']:i for i in load_selection(ROOT)}
    kinds=['legacy']+p['candidates'];aggregates=[];flat=[];latest=[]
    for h in p['horizons']:
        group=[t for t in tasks if t['horizon']==h];a={'horizon':h}
        for kind in kinds:
            values=[t['test_legacy'] if kind=='legacy' else t['test_candidates'][kind] for t in group]
            a[kind]={name:float(np.mean([m[name] for m in values])) for name in ('mae_pp','bias_pp','error_std_pp','rmse_pp')}
        aggregates.append(a)
    for t in tasks:
        for kind,m in t['test_candidates'].items():
            control=kind[:-3]+'_1y'
            flat.append(dict(code=t['code'],name=items[t['code']]['display_name'],horizon=t['horizon'],candidate=kind,**m,
                             matched_one_year_mae_pp=t['test_candidates'][control]['mae_pp'],legacy_mae_pp=t['test_legacy']['mae_pp'],
                             zero_mae_pp=t['test_zero']['mae_pp'],validation_mae_pp=t['validation_mean_mae_pp'][kind],choice=t['choice']))
        latest.append(dict(code=t['code'],name=items[t['code']]['display_name'],horizon=t['horizon'],choice=t['choice'],
                           **{kind+'_percent':v*100 for kind,v in t['latest_candidate_returns'].items()}))
    csvwrite(folder/'candidate_metrics.csv',flat);csvwrite(folder/'latest_candidate_predictions.csv',latest)
    with (folder/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    yearly=[]
    for t in tasks:
        group=[r for r in rows if r['code']==t['code'] and int(r['horizon'])==t['horizon']]
        for kind in p['candidates']:
            for year in sorted({r['date'][:4] for r in group}):
                subset=[r for r in group if r['candidate']==kind and r['date'].startswith(year)]
                yearly.append(dict(code=t['code'],name=items[t['code']]['display_name'],horizon=t['horizon'],candidate=kind,origin_year=year,
                                   **metrics([float(r['actual_return']) for r in subset],[float(r['expected_return']) for r in subset])))
    csvwrite(folder/'yearly_candidate_metrics.csv',yearly)
    bars,calendar=load_bars();datasets,_=samples(bars,p['codes'],calendar);coverage=[]
    for code in p['codes']:
        c=bars[code];first=datasets[1].loc[datasets[1].code==code,'date'].min()
        coverage.append(dict(code=code,name=items[code]['display_name'],raw_start=c.date.min(),raw_end=c.date.max(),
                             eligible_training_origin_start=first,latest_input_start=calendar[-1261],latest_input_end=calendar[-1],
                             past_sessions=1260,observed_closes=1261))
    csvwrite(folder/'history_coverage.csv',coverage)
    approved=sum(t['choice']!='legacy' for t in tasks)
    write(folder/'comparison.json',dict(run_id=s['run_id'],approved_upgrades=approved,aggregate_by_horizon=aggregates,coverage=coverage,
                                      interpretation='Exploratory previously observed test dates; five-year cycle hypothesis not established.'))
    lines=['# 五年历史输入与一年公平对照','',f'运行 `{s["run_id"]}`。8指数×5期限，4候选，固定300轮。','',
           '## 历史跨度的区别','',
           '原模型不是只用一年的样本训练：已有训练采用截至切分点的全部可用历史样本；但单次输入只覆盖252个交易日。此次把单次输入扩展为1260个交易日的变化，需要1261个连续收盘观测。按每年252交易日近似，实际起止日期见下表；不是严格五个公历年度。','',
           '五年版142个特征=原80个短期/联动特征+62个长周期特征：1/2/3/4/5年收益、波动、均线偏离、回撤、历史区间位置、相对成交量、20段季度收益及长期沪深300/A500相对强弱。它是五年行情的特征摘要，不是将1260根日线作为神经网络序列直接输入。没有按已知牛熊转折人为贴标签。','',
           '训练继续使用所有合格历史样本，未人为截掉五年前的训练样本。每个训练样本只能看到自身当日及之前五年的数据；标签结束日必须早于验证/测试起点。预测日指已收盘日，预测下一交易日或未来5/21/63/126交易日累计收益。','',
           '## 输入和样本覆盖','',
           '|方向|原始数据开始|最早五年输入样本日|本次最新输入起点|本次最新输入终点|','|---|---|---|---|---|']
    for c in coverage:lines.append('|'+ '|'.join(str(c[k]) for k in ('name','raw_start','eligible_training_origin_start','latest_input_start','latest_input_end'))+'|')
    lines+=['','各时点共享模型的训练起止、标签截止和行数保存于model.json的metadata；验证折覆盖保存于summary.json。原始数据早于最早可训练样本，是因为要先攒足五年及同行特征。A500发布前回溯和指数价格非基金总回报的限制仍存在。','',
            '## 公平对照及选择','',
            '一年/五年使用相同算法参数、相同训练起点集合、相同标签、相同验证/测试日期；一年对照仅使用原80特征，五年版使用142特征。因要求五年历史，两者均舍弃不具备完整五年输入的训练起点。当前旧模型保留原先更长的样本集合，不能把与旧模型的差异全部归因于输入窗口。','',
            '验证窗仍为2018—2019、2020—2021、2022—2023。只有五年候选有替换资格，相对旧模型、零预测、均值及两个一年对照中验证MAE最低者改善至少1%才采用；选择在测试前落盘。测试从2024年起，A500从2025年起，这些日期此前已经观察过，因此是探索性复测。',
            '',f'本轮通过替换门槛：**{approved}/40**。测试好坏不用于重新选择或调参。','']
    for metric,title in [('mae_pp','平均绝对误差MAE'),('error_std_pp','带符号误差标准差')]:
        lines += [f'## {title}（百分点，8指数等权平均）','', '|期限|'+'|'.join(LABELS[k] for k in kinds)+'|','|---|'+'---:|'*len(kinds)]
        for a in aggregates:lines.append('|'+TITLES[a['horizon']]+'|'+'|'.join(f'{a[k][metric]:.3f}' for k in kinds)+'|')
        lines+=['']
    lines+=['标准差为各指数带符号误差的总体标准差（ddof=0）再取8指数平均；均值、MAE和RMSE见CSV。长期标签相互重叠，8指数也相关，不能把这些观测都视为独立试验。','',
            '## 行业明细','', '|行业|交易日|当前模型MAE|LightGBM一年|LightGBM五年|CatBoost一年|CatBoost五年|验证选择|','|---|---:|---:|---:|---:|---:|---:|---|']
    for t in tasks:
        order=['lightgbm_mae_1y','lightgbm_mae_5y','catboost_mae_1y','catboost_mae_5y']
        lines.append(f'|{items[t["code"]]["display_name"]}|{t["horizon"]}|{t["test_legacy"]["mae_pp"]:.3f}|'+
                     '|'.join(f'{t["test_candidates"][k]["mae_pp"]:.3f}' for k in order)+'|'+t['choice']+'|')
    lines+=['','## 文件与复现','',
            '- [行业与期限全部指标](candidate_metrics.csv)：误差均值、标准差、MAE、RMSE。',
            '- [分年指标](yearly_candidate_metrics.csv)、[最新原始候选输出](latest_candidate_predictions.csv)、[逐日测试预测](test_predictions.csv)。',
            '- [输入与训练历史覆盖](history_coverage.csv)、[完整验证分数与训练折跨度](summary.json)、[独立重放验证](verification.json)。',
            '- `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python code/training/train_five_year.py`：固定协议训练，相同运行不覆盖。',
            '- `.venv/bin/python code/training/verify_five_year.py`：独立重载，不拟合。',
            '- `.venv/bin/python code/training/report_five_year.py`：生成本报告。','',
            '五年输入只提供更长背景，不保证存在固定3—5年牛熊周期，也不保证预测改善；本轮不改变Kronos的下载阻塞状态。','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(folder/'README.md',flush=True)
    for a in aggregates:print(TITLES[a['horizon']],{k:round(a[k]['mae_pp'],3) for k in kinds},flush=True)
    print('APPROVED',approved,flush=True)


if __name__=='__main__':report()

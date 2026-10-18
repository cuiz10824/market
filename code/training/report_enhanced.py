"""Compare all frozen OHLCV candidates, including unsuccessful experiments."""
import csv
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from daily.selection import load_selection
from training.train_enhanced import read,write,metrics

TITLES={1:'下一交易日',5:'一周',21:'一月',63:'三月',126:'六月'}


def csvwrite(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report():
    pointer=read(ROOT/'code/models/latest_enhanced.json');summary=read(ROOT/pointer['report'])
    folder=(ROOT/pointer['report']).parent;p=summary['protocol'];tasks=summary['tasks']
    items={i['code']:i for i in load_selection(ROOT)}
    aggregates=[];flat=[];latest=[]
    for h in p['horizons']:
        group=[t for t in tasks if t['horizon']==h]
        result={'horizon':h}
        for kind in ['legacy','zero']+p['candidates']:
            ms=[t['test_'+kind] if kind in ('legacy','zero') else t['test_candidates'][kind] for t in group]
            result[kind]={name:float(np.mean([m[name] for m in ms])) for name in ('mae_pp','bias_pp','error_std_pp','rmse_pp')}
            result[kind]['better_than_legacy']=sum(m['mae_pp']<t['test_legacy']['mae_pp']-1e-12 for t,m in zip(group,ms))
        aggregates.append(result)
    for task in tasks:
        for kind,m in task['test_candidates'].items():
            flat.append({'code':task['code'],'name':items[task['code']]['display_name'],'horizon':task['horizon'],
                         'candidate':kind,**m,'legacy_mae_pp':task['test_legacy']['mae_pp'],
                         'zero_mae_pp':task['test_zero']['mae_pp'],
                         'mae_improvement_vs_legacy_pp':task['test_legacy']['mae_pp']-m['mae_pp'],
                         'validation_choice':task['choice'],'validation_mae_pp':task['validation_mean_mae_pp'][kind]})
        latest.append({'code':task['code'],'name':items[task['code']]['display_name'],'horizon':task['horizon'],
                       'validation_choice':task['choice'],
                       **{kind+'_return_percent':value*100 for kind,value in task['latest_candidate_returns'].items()}})
    csvwrite(folder/'candidate_metrics.csv',flat);csvwrite(folder/'latest_candidate_predictions.csv',latest)
    with (folder/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    yearly=[]
    for task in tasks:
        group=[r for r in records if r['code']==task['code'] and int(r['horizon'])==task['horizon']]
        for kind in p['candidates']:
            for year in sorted({r['date'][:4] for r in group}):
                rows=[r for r in group if r['candidate']==kind and r['date'].startswith(year)]
                actual=[float(r['actual_return']) for r in rows]
                m=metrics(actual,[float(r['expected_return']) for r in rows])
                old=metrics(actual,[float(r['legacy_return']) for r in rows])
                yearly.append({'code':task['code'],'name':items[task['code']]['display_name'],'horizon':task['horizon'],
                               'candidate':kind,'origin_year':year,**m,'legacy_mae_pp':old['mae_pp']})
    csvwrite(folder/'yearly_candidate_metrics.csv',yearly)
    write(folder/'comparison.json',{'run_id':summary['run_id'],'aggregate_by_horizon':aggregates,
          'approved_upgrades':sum(t['choice']!='legacy' for t in tasks),'kronos_status':read(ROOT/'data/reports/kronos_mini/status.json')})
    lines=['# OHLCV、CatBoost、改进LightGBM与Kronos实验','',
           f'运行 `{summary["run_id"]}`；当前8指数×5期限。原模型、数据、历史报告未覆盖。','',
           '## 本轮实际完成','',
           '- 从原始腾讯响应整理并校验36,857条开高低收与成交量，8个指数在各自覆盖期无交易日缺口；收盘与旧库匹配。新库data/enhanced/market_ohlcv.sqlite3，旧库不改。',
           '- 使用80个已知历史特征：OHLC形态、短中长期收益与波动、相对成交量、市场宽度、沪深300/A500联动、行业相对强弱、8个指数身份编码。没有使用未来价格、未发布宏观或未核验估值。',
           '- 成交量保留腾讯原始单位，仅使用序列内比值；未把字段不清的成交额当已核验数据。指数回溯尤其A500仍是历史局限。',
           '- 训练4种候选：LightGBM-MAE、LightGBM-MSE、CatBoost-MAE、CatBoost-RMSE。每种使用固定300轮、树深5；每期限共享8指数训练数据，保留指数身份和交叉行业特征。',
           '- 使用2018—2019、2020—2021、2022—2023三个扩展验证窗。按标签结束日期在所有指数上隔离训练和验证，不在测试集上早停。',
           '- 测试从2024年起，A500从2025年起，分别保存对应时点模型。预测日期和真实收益与旧报告逐条对齐；当前表是已观察历史区间的探索性复测，不是新的独立未来成绩。',
           '- 测试后按同样固定参数重训latest，保存全部候选，便于查看其非零输出。不能把holdout成绩作为latest的新独立成绩。','',
           '## 预先固定的替换规则','',
           '每指数、每期限仅按滚动验证的MAE选择新候选；相对旧模型验证成绩、零预测、历史均值三者较好值改善至少1%才替换。测试成绩不改变选择。本轮通过门槛 '+str(sum(t['choice']!='legacy' for t in tasks))+'/40 组，因此日常输出沿用验证选择，未强制产生非零预测。','',
           '## 测试平均绝对误差 MAE（百分点，8指数等权）','',
           '|期限|旧模型|零预测|LightGBM-MAE|LightGBM-MSE|CatBoost-MAE|CatBoost-RMSE|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for a in aggregates:
        lines.append('|'+TITLES[a['horizon']]+'|'+'|'.join(f'{a[k]["mae_pp"]:.3f}' for k in ['legacy','zero']+p['candidates'])+'|')
    lines+=['','## 测试误差标准差（百分点，8指数等权）','',
            '|期限|旧模型|零预测|LightGBM-MAE|LightGBM-MSE|CatBoost-MAE|CatBoost-RMSE|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for a in aggregates:
        lines.append('|'+TITLES[a['horizon']]+'|'+'|'.join(f'{a[k]["error_std_pp"]:.3f}' for k in ['legacy','zero']+p['candidates'])+'|')
    lines+=['','标准差按每个指数的带符号误差计算（ddof=0），再对8个指数平均；误差均值与RMSE见CSV，标准差不能替代偏差和MAE。','',
            '## 各任务的测试MAE（百分点）','',
            '|方向|交易日|旧模型|LightGBM-MAE|LightGBM-MSE|CatBoost-MAE|CatBoost-RMSE|验证选择|',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    for t in tasks:
        lines.append('|'+items[t['code']]['display_name']+'|'+str(t['horizon'])+f'|{t["test_legacy"]["mae_pp"]:.3f}|'+
                     '|'.join(f'{t["test_candidates"][k]["mae_pp"]:.3f}' for k in p['candidates'])+'|'+t['choice']+'|')
    status=read(ROOT/'data/reports/kronos_mini/status.json')
    lines+=['','## Kronos-mini','',
            '官方源码按提交67b630e67f6a18c9e9be918d9b4337c960db1e9a固定，MIT许可；CPU PyTorch及其依赖已安装，模型模块导入通过。',
            '',f'实际状态：**{status["status"]}**。'+('已完成稀疏起点诊断，详见状态JSON。' if status.get('tested') else '访问官方Hugging Face模型元数据时被代理CONNECT 403拒绝，未取得预训练权重，未训练或推理，不能提供性能数字。'),
            '', '可执行入口：`python code/training/benchmark_kronos.py`。缓存官方权重、固定修订与SHA256后，以256个历史交易日、4次采样、每21日一个预测起点比较1/5/21/63/126日收益；起点从2025-09-01开始。缺失成交额通道明确置为缺失占位0，不作为真实观测。预训练截止日期尚未核验，结果只作诊断，不自动上网页。',
            '', '## 复现与文件','',
            '- `python data/tools/build_ohlcv.py`：原始快照→新OHLCV数据库与分类CSV。',
            '- `.venv/bin/python code/training/train_enhanced.py`：固定协议训练；相同指纹不覆盖既有运行。',
            '- `.venv/bin/python code/training/verify_enhanced.py`：不训练，重载全部原生模型并核对保存预测、来源SHA和旧模型不变性。',
            '- `.venv/bin/python code/training/report_enhanced.py`：生成本对比报告。',
            '- [160组候选指标](candidate_metrics.csv)：均值、MAE、标准差、RMSE及旧/零基准比较。',
            '- [分年候选指标](yearly_candidate_metrics.csv)：按预测起点年份，避免仅看总体均值。',
            '- [候选当前非零输出](latest_candidate_predictions.csv)：每个指数期限的四个原始候选预测，仅供研究，未通过验证门槛的不会替换日常结果。',
            '- [全部候选逐日预测](test_predictions.csv)、[完整验证分数](summary.json)、[汇总](comparison.json)。','',
            '多日标签重叠、跨指数相关、预先已观察测试价格和未审计的回溯历史仍限制结论。本轮失败或改善都完整保留，不按最终测试重新挑参数。','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(folder/'README.md')
    for a in aggregates:print(TITLES[a['horizon']],{k:round(a[k]['mae_pp'],3) for k in ['legacy']+p['candidates']})


if __name__=='__main__':report()

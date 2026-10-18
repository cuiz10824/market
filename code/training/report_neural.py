"""Summarize all shared neural candidates without selecting on held-out scores."""
import csv
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from training.train_enhanced import read,write,metrics
from training.report_enhanced import csvwrite,TITLES
from daily.selection import load_selection
from research.sequence_data import SequenceData

LABELS={'legacy':'当前模型','zero':'零预测','validation_selected':'验证所选组合','dlinear':'DLinear','nlinear':'NLinear','tcn':'小TCN','patchtst':'小PatchTST'}


def report():
    pointer=read(ROOT/'code/models/latest_neural.json');s=read(ROOT/pointer['report']);p=s['protocol']
    folder=(ROOT/pointer['report']).parent;tasks=s['tasks'];index=read(ROOT/pointer['index'])
    items={i['code']:i for i in load_selection(ROOT)};aggregates=[];flat=[];latest=[]
    kinds=['legacy','zero','validation_selected']+p['architectures']
    for h in p['horizons']:
        group=[t for t in tasks if t['horizon']==h];result={'horizon':h}
        for kind in kinds:
            if kind=='validation_selected':
                values=[t['test_legacy'] if t['choice']=='legacy' else t['test_candidates'][t['choice']] for t in group]
            else:values=[t['test_'+kind] if kind in ('legacy','zero') else t['test_candidates'][kind] for t in group]
            result[kind]={name:float(np.mean([m[name] for m in values])) for name in ('mae_pp','bias_pp','error_std_pp','rmse_pp')}
        aggregates.append(result)
    for t in tasks:
        for a,m in t['test_candidates'].items():
            flat.append(dict(code=t['code'],name=items[t['code']]['display_name'],horizon=t['horizon'],candidate=a,
                             lookback=s['chosen_lookbacks'][a],**m,legacy_mae_pp=t['test_legacy']['mae_pp'],zero_mae_pp=t['test_zero']['mae_pp'],
                             validation_mae_pp=t['validation_mean_mae_pp'][a],research_eligible_choice=t['choice']))
        latest.append(dict(code=t['code'],name=items[t['code']]['display_name'],horizon=t['horizon'],research_eligible_choice=t['choice'],
                           **{a+'_percent':v*100 for a,v in t['latest_candidate_returns'].items()}))
    csvwrite(folder/'candidate_metrics.csv',flat);csvwrite(folder/'latest_candidate_predictions.csv',latest)
    selected=[]
    for t in tasks:
        score=t['test_legacy'] if t['choice']=='legacy' else t['test_candidates'][t['choice']]
        selected.append(dict(code=t['code'],name=items[t['code']]['display_name'],horizon=t['horizon'],choice=t['choice'],**score,
                             legacy_mae_pp=t['test_legacy']['mae_pp'],mae_improvement_pp=t['test_legacy']['mae_pp']-score['mae_pp'],
                             relative_validation_gain=t['relative_validation_gain']))
    csvwrite(folder/'selected_metrics.csv',selected)
    with (folder/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    yearly=[];seeds=[]
    for t in tasks:
        group=[r for r in rows if r['code']==t['code'] and int(r['horizon'])==t['horizon']]
        for a in p['architectures']:
            selected=[r for r in group if r['candidate']==a]
            for year in sorted({r['date'][:4] for r in selected}):
                subset=[r for r in selected if r['date'].startswith(year)]
                actual=[float(r['actual_return']) for r in subset]
                yearly.append(dict(code=t['code'],name=items[t['code']]['display_name'],horizon=t['horizon'],candidate=a,origin_year=year,
                                   **metrics(actual,[float(r['expected_return']) for r in subset]),
                                   legacy_mae_pp=metrics(actual,[float(r['legacy_return']) for r in subset])['mae_pp']))
            for member,column in [('seed1','seed1_return'),('seed2','seed2_return'),('ensemble','expected_return')]:
                seeds.append(dict(code=t['code'],name=items[t['code']]['display_name'],horizon=t['horizon'],candidate=a,member=member,
                                  **metrics([float(r['actual_return']) for r in selected],[float(r[column]) for r in selected])))
    csvwrite(folder/'yearly_candidate_metrics.csv',yearly);csvwrite(folder/'seed_metrics.csv',seeds)
    eligible=sum(t['choice']!='legacy' for t in tasks)
    write(folder/'comparison.json',dict(run_id=s['run_id'],aggregate_by_horizon=aggregates,research_eligible_tasks=eligible,
                                      chosen_lookbacks=s['chosen_lookbacks'],refit_epochs=s['refit_epochs'],auto_deploy=False))
    data=SequenceData(p);coverage=[]
    for ci,c in enumerate(p['codes']):
        ix=data.ids==ci
        coverage.append(dict(code=c,name=items[c]['display_name'],eligible_origin_start=min(data.dates[ix]),origin_end=max(data.dates[ix]),
                             available_origins=int(ix.sum()),latest_1260_input_start=str(data.calendar[-1261]),latest_as_of=str(data.calendar[-1])))
    csvwrite(folder/'history_coverage.csv',coverage)
    fits=[]
    for e in index['models']:
        m=read(ROOT/e['bundle'])['metadata']
        fits.append(dict(role=e['role'],architecture=e['architecture'],lookback=e['lookback'],seed=e['seed'],cutoff=e['start'],
                         training_rows=m['training_rows'],fit_origin_start=m['fit_origin_start'],fit_origin_end=m['fit_origin_end'],
                         selected_epoch=m['selected_epoch'],epochs_executed=m['epochs_executed'],parameter_count=m['parameter_count'],
                         **{'label_end_'+str(h):m['fit_label_ends'][str(h)] for h in p['horizons']}))
    csvwrite(folder/'fit_coverage.csv',fits)
    lines=['# 共享小型神经网络实验','',f'运行 `{s["run_id"]}`。四种结构、8个指数、五个直接累计收益输出；独立研究实验，未更改每日模型。','',
           '## 实际训练与时间隔离','',
           f'- 使用既有36,857条OHLCV。共有{s["origin_count"]:,}个满足最长1260交易日输入的合格起点，包含最新无完整未来标签的起点；实际训练行数及逐头标签截止见fit_coverage.csv。',
           '- 日线按从预测日向前对齐的5交易日聚合，第一组可以短于5日。6通道：相对当前收盘的对数收盘、开盘跳空、高低范围、组内收益、相对日均成交量、整组收益。只使用当时已知值，量使用同指数窗口比值。没有把缺日补成虚假行情。',
           '- 单次输入比较252/756/1260交易日，分别得到51/152/252个分组。三种长度共享同一批满足1260+1连续观测的起点，训练按观测日历每5日取样，测试与最终验证指标逐日计算。',
           '- 所有结构跨8指数共享参数，加入指数ID偏置，直接输出未来1/5/21/63/126交易日价格收益；没有把126次单日递推当半年预测。',
           '- 五个头的标签单独掩码：每个训练标签结束日严格早于切分点，未知标签不参与损失。最新训练仍能使用近期已知的短期标签，而不伪造尚未发生的半年收益。',
           '- 损失为按固定0.01×sqrt(h)缩放后的各头平均绝对误差；此缩放不是从未来拟合的市场波动。最终指标全部还原成累计收益百分点。',
           '- 验证窗2018—2019、2020—2021、2022—2023；第一个种子比较长度，第二个种子复现选中长度，两者等权平均。仅用验证选择早停轮数，测试重训按六个验证模型所选轮数的中位数取整。',
           '- 每次最多12轮、最少2轮、耐心3轮，AdamW、固定学习率0.001和小容量网络。训练只用CPU。这是有限预算实验，不能代表各结构最优性能。',
           '- 测试从2024年起，A500从2025年起，与旧报告预测日期和真实收益逐项对齐。这些价格区间此前已观察过，属于探索性复测，不是全新独立未来测试。','',
           '## 网络与验证选中的配置','',
           '|结构|输入交易日|分组数|测试/latest训练轮数|参数量/单个网络|','|---|---:|---:|---:|---:|']
    for a in p['architectures']:
        e=next(e for e in fits if e['architecture']==a and e['role']=='latest')
        lines.append(f'|{LABELS[a]}|{s["chosen_lookbacks"][a]}|{(s["chosen_lookbacks"][a]+4)//5}|{s["refit_epochs"][a]}|{e["parameter_count"]:,}|')
    lines+=['','DLinear/NLinear是改为收益输出头的轻量基线；TCN为8通道、8个因果扩张卷积残差块；PatchTST为8维、2注意力头、1层编码器，按8个分组（约40交易日）切patch，各通道共用编码器并学习通道混合。Patch不足时只复制首个已知分组作网络边界padding，没有增加日线记录。它们是参考结构后的项目适配实现，不是原论文基准的完全复现，不使用外部预训练权重。','',
            '全部3种长度的验证成绩和全部拟合日志均保留；每种结构只对验证选中的长度做最终测试，因此不能把此表称为三种长度全部测试的成绩。单个种子与集成的误差见seed_metrics.csv，不能据两个种子推断稳定显著优势。','']
    for key,title in [('mae_pp','平均绝对误差MAE'),('error_std_pp','带符号误差标准差')]:
        lines += [f'## {title}（百分点，8指数等权平均）','', '|期限|'+'|'.join(LABELS[k] for k in kinds)+'|','|---|'+'---:|'*len(kinds)]
        for a in aggregates:lines.append('|'+TITLES[a['horizon']]+'|'+'|'.join(f'{a[k][key]:.3f}' for k in kinds)+'|')
        lines+=['']
    lines+=['标准差采用ddof=0，先逐指数计算带符号误差标准差，再取8指数平均。偏差均值和RMSE在CSV中；不能只看误差标准差忽略系统性偏差。','',
            '## 研究资格与每日模型','',
            f'按验证MAE预先固定的≥1%改善门槛，共 **{eligible}/40** 任务获得研究替换资格。比较基准取当前模型验证成绩、零预测、相同训练采样历史均值中较好者。每组的选择在测试前写入choices.json，未按测试重新选择。',
            '', '“验证所选组合”逐任务采用测试前冻结的选择：获批任务采用相应神经网络，其余任务沿用旧模型；它不是测试集上选最优的结果。各单一结构列则对所有40个任务都用该结构，二者用途不同。',
            '', '本轮权重和原始最新输出保存在独立neural目录；latest_neural.json标记auto_deploy=false，不改变网页或Windows依赖。实际每日服务仍采用已有验证选择。获得研究资格也不等于已证明独立未来优势或已上线。','',
            '## 行业明细','', '|行业|交易日|当前模型MAE|DLinear|NLinear|小TCN|小PatchTST|验证研究选择|','|---|---:|---:|---:|---:|---:|---:|---|']
    for t in tasks:
        lines.append(f'|{items[t["code"]]["display_name"]}|{t["horizon"]}|{t["test_legacy"]["mae_pp"]:.3f}|'+
                     '|'.join(f'{t["test_candidates"][a]["mae_pp"]:.3f}' for a in p['architectures'])+'|'+t['choice']+'|')
    lines+=['','## 文件和复现','',
            '- [160组行业期限指标](candidate_metrics.csv)、[40组验证所选结果](selected_metrics.csv)、[分年指标](yearly_candidate_metrics.csv)、[两种子与集成指标](seed_metrics.csv)。',
            '- [全部逐日测试预测](test_predictions.csv)、[原始最新候选输出](latest_candidate_predictions.csv)、[拟合时间覆盖](fit_coverage.csv)、[输入历史覆盖](history_coverage.csv)。',
            '- [完整验证与结果](summary.json)、[输入长度筛选](lookback_choices.json)、[测试前冻结的选择](choices.json)、[独立重放验证](verification.json)。',
            '- `.venv/bin/python code/training/train_neural.py`：按冻结协议训练，相同指纹拒绝覆盖。',
            '- `.venv/bin/python code/training/verify_neural.py`：不训练，安全重载全部safetensors并重放预测、标签、掩码和选择。',
            '- `.venv/bin/python code/training/report_neural.py`：生成本报告。','',
            '历史回溯尤其A500、指数价格非基金含分红收益、相关指数与重叠长期标签仍限制结论；本轮未做统计显著性宣称或块式置信区间。首轮导出曾因NumPy字符串日期归约兼容性失败，在首个模型保存前终止，失败记录单列保留，不计入性能。','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(folder/'README.md')
    for a in aggregates:print(TITLES[a['horizon']],{k:round(a[k]['mae_pp'],3) for k in kinds})
    print('RESEARCH_ELIGIBLE',eligible)


if __name__=='__main__':report()

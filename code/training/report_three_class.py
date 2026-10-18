"""Human-readable comparison of the fixed three-class experiment."""
import csv
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from training.train_three_class import read
from training.three_class import metrics


def report():
    pointer=read(ROOT/'code/models/latest_three_class.json')
    index=read(ROOT/pointer['index'])
    path=ROOT/index['report']
    summary=read(path)
    tasks=summary['tasks']
    lines=['# ±5%三分类实验报告','',
           '规则：未来累计价格收益 < −5% 为大跌，−5% 至 +5%（含边界）为波动，> +5% 为大涨。所有期限使用相同阈值。',
           '', '原来的数值回归模型、参数、预测和数据库保持不变，33,710 条回归预测逐条完全一致。',
           '', '## 评估方法与限制','',
           '- 14 个指数 × 5、21、63、126 个交易日，共 56 个任务。周/月/季/半年均为交易日近似。',
           '- 只在 2018—2023 年的三个滚动验证窗口选择模型；训练标签结束日期必须早于验证/测试开始日期。',
           '- 比较历史类别频率、始终波动、逻辑回归、LightGBM，以及两种类别均衡加权模型。以三类等权宏平均 F1 选型；复杂模型增益低于 0.005 时保留基线。',
           '- 测试期 2024 年起，中证 A500 为 2025 年起。该区间此前已经查看，本次是探索性复测，并非新的独立样本外证明。参数未根据本次测试成绩调整。',
           '- 相邻未来收益窗口重叠，特别是半年期有效独立样本很少，不能把每日样本数当成独立样本量。',
           '- 三分类与原二分类的准确率不能直接比较；短期波动类占比大，高准确率可能来自始终预测波动。概率未做校准，不能作为可靠置信度。',
           '- 价格收益不含分红、费用。A500 含指数发布前回溯历史；医药/医疗部分历史较短；长期结论仍受数据与市场阶段限制。',
           '', '## 按期限汇总','',
           '准确率和 F1 为 14 个指数等权平均；类别占比按该期限所有测试行汇总。F1 越高越好，兼顾三类。','',
           '|期限|实际大跌/波动/大涨占比|模型准确率|始终波动准确率|模型宏 F1|始终波动宏 F1|历史频率宏 F1|',
           '|---|---|---:|---:|---:|---:|---:|']
    for h,title in [(5,'一周'),(21,'一月'),(63,'三月'),(126,'六月')]:
        a=summary['aggregate_by_horizon'][str(h)]
        counts=a['actual_counts'];total=sum(counts)
        proportions='/'.join(f'{v/total:.1%}' for v in counts)
        lines.append(f'|{title}|{proportions}|{a["selected"]["accuracy"]:.1%}|{a["always_neutral"]["accuracy"]:.1%}|{a["selected"]["macro_f1"]:.3f}|{a["always_neutral"]["macro_f1"]:.3f}|{a["historical_frequency"]["macro_f1"]:.3f}|')
    beat=sum(t['test']['selected']['macro_f1']>max(t['test'][k]['macro_f1'] for k in ['always_neutral','historical_frequency'])+1e-12 for t in tasks)
    lines+=['',f'56 组中，{beat} 组测试宏 F1 超过两种基线的较好者。这是描述性比较，不能证明稳定超额预测能力。','',
            '## 各行业、各期限','',
            '召回率表示真实属于该类时有多少被识别出来，顺序为“大跌 / 波动 / 大涨”。','',
            '|指数|交易日|模型|准确率|宏 F1|历史频率 F1|始终波动 F1|三类召回率|原回归 MAE（百分点）|',
            '|---|---:|---|---:|---:|---:|---:|---|---:|']
    flat=[]
    for t in tasks:
        test=t['test'];m=test['selected'];recalls='/'.join(f'{c["recall"]:.1%}' for c in m['per_class'])
        lines.append(f'|{t["name"]}|{t["horizon"]}|{t["choice"]}|{m["accuracy"]:.1%}|{m["macro_f1"]:.3f}|{test["historical_frequency"]["macro_f1"]:.3f}|{test["always_neutral"]["macro_f1"]:.3f}|{recalls}|{test["unchanged_regression"]["mae"]*100:.2f}|')
        flat.append({'code':t['code'],'name':t['name'],'horizon':t['horizon'],'model':t['choice'],
                     'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
                     'frequency_macro_f1':test['historical_frequency']['macro_f1'],
                     'neutral_macro_f1':test['always_neutral']['macro_f1'],
                     **{f'{label}_recall':m['per_class'][i]['recall'] for i,label in enumerate(['down','neutral','up'])},
                     'unchanged_regression_mae':test['unchanged_regression']['mae']})
    lines+=['','## 混淆矩阵','', '行是真实类别，列是预测类别；两者顺序均为大跌、波动、大涨。', '']
    for t in tasks:
        lines += [f'### {t["name"]} / {t["horizon"]} 交易日','',
                  '|真实\\预测|大跌|波动|大涨|','|---|---:|---:|---:|']
        for label,row in zip(['大跌','波动','大涨'],t['test']['selected']['confusion_matrix']):
            lines.append('|'+label+'|'+'|'.join(map(str,row))+'|')
        lines.append('')
    lines+=['## 文件和复现','',
            '- `summary.json`：验证选型、每类精确率/召回率/F1、混淆矩阵和基线的完整结果。',
            '- `task_metrics.csv`：56 组任务比较；`yearly_metrics.csv`：按预测起始年份分组，不能用于再次调参。',
            '- `test_predictions.csv`：逐条三类概率、分类、真实收益和保持原样的回归预测。',
            '- `verification.json`：新模型及全部原始模型、原始报告、数据库的 SHA256。',
            '- 最新模型指针为 `code/models/latest_three_class.json`；原 `latest.json` 保留。',
            '- 仓库根目录执行 `.venv/bin/python code/training/verify_three_class.py` 复验。','']
    (path.parent/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    def csvwrite(name,rows):
        with (path.parent/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csvwrite('task_metrics.csv',flat)
    with (path.parent/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        predictions=list(csv.DictReader(f))
    yearly=[]
    for t in tasks:
        group=[r for r in predictions if r['code']==t['code'] and int(r['horizon'])==t['horizon']]
        for year in sorted({r['date'][:4] for r in group}):
            rows=[r for r in group if r['date'].startswith(year)]
            m=metrics([int(r['actual_class']) for r in rows],np.array([[float(r[k]) for k in ['p_big_down','p_neutral','p_big_up']] for r in rows]))
            yearly.append({'code':t['code'],'name':t['name'],'horizon':t['horizon'],'origin_year':year,
                           'n':len(rows),'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
                           **{f'{label}_recall':m['per_class'][i]['recall'] for i,label in enumerate(['down','neutral','up'])}})
    csvwrite('yearly_metrics.csv',yearly)
    print(path.parent/'README.md')


if __name__=='__main__':report()

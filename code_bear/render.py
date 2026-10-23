"""Export a portable Chinese report and standalone scientific charts."""
import base64
import html
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/code-bear-matplotlib' if os.name!='nt' else str(Path(__file__).parent/'.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as md
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
import numpy as np
import pandas as pd

from train import ROOT,read
from bear.dataset import cycles,NAMES

MODEL_NAMES={'majority':'多数类基准','persistence':'上一同期限趋势延续','ma_rule':'均线趋势规则',
             'logistic':'多项逻辑回归','lightgbm':'LightGBM'}
HORIZONS={63:'未来约3个月',126:'未来约半年',252:'未来约一年'}


def draw(frame,legs,path,years=None):
    data=frame.copy();data['date']=pd.to_datetime(data.date)
    for w in (120,250):data[f'ma{w}']=data.close.rolling(w).mean()
    if years:data=data[data.date>=data.date.max()-pd.DateOffset(years=years)]
    fig,(ax,vol)=plt.subplots(2,1,figsize=(15,6.3),sharex=True,gridspec_kw={'height_ratios':[4,1]})
    for row in legs.itertuples():
        start=max(pd.Timestamp(row.start),data.date.min());end=min(pd.Timestamp(row.end),data.date.max())
        if start>=end:continue
        ax.axvspan(start,end,color=('#c83e3e' if row.direction=='bull' else '#15835d') if row.confirmed else '#8e96a3',
                   alpha=.10,hatch=None if row.confirmed else '//',linewidth=0)
    if years:
        monthly=data.assign(observed=data.date).set_index('date').resample('ME').agg(
            dict(open='first',high='max',low='min',close='last',volume='sum',ma120='last',ma250='last',observed='last')).dropna(subset=['close'])
        for row in monthly.itertuples():
            x=md.date2num(row.observed);color='#c83e3e' if row.close>=row.open else '#15835d'
            ax.vlines(x,row.low,row.high,color=color,linewidth=.8)
            ax.add_patch(Rectangle((x-7,min(row.open,row.close)),14,max(abs(row.close-row.open),2),facecolor=color,edgecolor=color,linewidth=.6))
            vol.bar(x,row.volume/1e8,width=16,color=color)
        ax.plot(monthly.observed,monthly.ma120,color='#e79d19',lw=1,label='Daily MA120')
        ax.plot(monthly.observed,monthly.ma250,color='#5269d9',lw=1,label='Daily MA250')
        ax.set_title('Shanghai Composite: monthly OHLC from daily observations (last month partial)',loc='left',fontsize=12)
        ax.legend(loc='upper left',fontsize=8)
    else:
        ax.plot(data.date,data.close,color='#333d4a',lw=.8)
        vol.bar(data.date,data.volume/1e8,width=1,color='#707982',alpha=.5)
        ax.set_title('Shanghai Composite: observed daily close, 1990 onward',loc='left',fontsize=12)
    ax.set_ylabel('Index points');vol.set_ylabel('Volume\n(source units /1e8)',fontsize=8)
    ax.grid(axis='y',alpha=.2);vol.grid(axis='y',alpha=.15)
    ax.spines[['top','right']].set_visible(False);vol.spines[['top','right']].set_visible(False)
    vol.xaxis.set_major_locator(md.YearLocator(1 if years else 4));vol.xaxis.set_major_formatter(md.DateFormatter('%Y'))
    fig.legend(handles=[Patch(color='#c83e3e',alpha=.2,label='Bull leg, HINDSIGHT'),
                        Patch(color='#15835d',alpha=.2,label='Bear leg, HINDSIGHT'),
                        Patch(facecolor='#ccc',hatch='//',label='Last leg unconfirmed')],loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(path,dpi=135,facecolor='white');plt.close(fig)


def render():
    pointer=read(ROOT/'models/latest.json');s=read(ROOT/pointer['report']);folder=(ROOT/pointer['report']).parent
    current=read(ROOT/'output/current_prediction.json');data=current['dataset'];manifest=read(ROOT/data['manifest'])
    frame=pd.read_csv(ROOT/data['csv']);legs=cycles(frame,s['protocol']['retrospective_reversal'])
    output=ROOT/'output';output.mkdir(exist_ok=True)
    legs.to_csv(output/'historical_cycles.csv',index=False,lineterminator='\n')
    for name,years in [('kline_10y.png',10),('history_cycles.png',None)]:draw(frame,legs,output/name,years)
    table=['|期限|验证所选方案|所选测试准确率|所选测试Macro-F1|多数类Macro-F1|LightGBM测试Macro-F1|',
           '|---|---|---:|---:|---:|---:|']
    for h in s['protocol']['horizons']:
        kind=s['choices'][str(h)]['candidate'];m=s['test_scores'][str(h)]
        table.append(f'|{HORIZONS[h]}|{MODEL_NAMES[kind]}|{m[kind]["accuracy"]:.3f}|{m[kind]["macro_f1"]:.3f}|{m["majority"]["macro_f1"]:.3f}|{m["lightgbm"]["macro_f1"]:.3f}|')
    model_count=len(read(ROOT/pointer['index'])['models'])
    lines=['# 上证指数牛熊趋势：第一轮实验','',f'运行 `{s["run_id"]}`。数据{s["rows"]}行（本次训练），截至{s["last_date"]}；特征起点{s["first_feature_date"]}，共{s["feature_count"]}个过去数据特征。','',
        '## 第一轮结论','',
        '历史大波段可以清楚描述，但当前结果尚未证明能稳定提前预测牛熊转折。验证选中的3个月、半年方案都是“上一同期限趋势延续”基准；一年选中逻辑回归。LightGBM一年测试Macro-F1较高，但验证阶段未胜出，不能看完测试后改选它。','',
        '## 目标与避免事后信息','',
        '- 未来63/126/252交易日累计价格收益分别至少+6%/+10%/+15%，且至少60%未来收盘价高于预测日收盘价，记为牛市趋势；对称负收益阈值且至少60%低于起点记为熊市趋势，其余为震荡/混合。是本项目的趋势定义，不是官方牛熊认定。',
        '- 末63/126/252日起点尚未走完整个期限，相应真实标签留空，不补成震荡。当前预测可产生，但不能现在检验。',
        '- 20%反转确认的历史波段只作事后图示；拐点发生日与确认日分别保存，末段未确认。绝不把这类事后波段当作当日已知输入。',
        '- 输入只有当日及过去开高低收和成交量：收益、均线位置与斜率、回撤、波动、相对成交量等，最长需要1270个已完成交易日（约五年）。历史训练样本扩展使用更早多年，未假设固定3—5年必然重复。',
        '- 验证窗2010—2013、2014—2017、2018—2021；训练标签结束日严格早于切分点，验证标签也限制在验证窗内。每期限只按三折平均Macro-F1选方案。',
        '- 测试按2022/2023/2024/2025/2026年依次重训，每次只能使用年初之前已经落地的标签。测试起点属于该年，真实结果允许延续至次年；尚未落地的不评分。图中的年份已被观察，因此是探索性历史对照，不是封存的未来测试。',
        '- 标准化只在训练数据拟合，超参数事先固定。标签重叠且牛熊周期数有限，日样本数不等于独立周期数；本轮没有交易收益或显著性结论。','',
        '## 时间顺序测试','',*table,'',
        'Macro-F1同时考虑三类，越高越好。震荡占比较大，多数类准确率可以较高，因此同时报告平衡准确率、各类样本数、混淆矩阵、Log-loss和Brier；详情见summary.json及yearly_metrics.csv。年度内缺某一类时平衡准确率只在真实存在的类别上定义，不能当成三类都识别准确。', '',
        '## 已保存资料','',
        f'- {model_count}份JSON/LightGBM原生模型及分割预测，原生重载检查与独立复验见verification.json。',
        '- test_predictions.csv：各模型每个测试起点的实际趋势、预测类别、类别分数和实际未来收益。',
        '- historical_cycles.csv：历史20%波段的起止、方向、确认日期、末段是否确认。past_only_trend.csv：过去数据均线规则状态；早期均线热身不用于模型训练或测试。',
        '- 数据manifest.json记录来源、37份压缩原始响应、重叠校验、日历覆盖、单位限制。2000年起的旧观测交易日历6489天全部存在；1990年代及旧日历以后的日期主要依赖同源观测，不能称全部独立跨源核验。',
        '- 成交量保留腾讯原始单位，只用同源相对量特征；未独立确认换算单位，没有伪造成交额。指数价格不含基金分红及跟踪误差。', '',
        '网页code_bear/index.html可在Windows直接打开；图为历史资料，未来卡片为已选方法在最新日线上的输出。规则没有概率；学习模型的类别分数未经校准，不称为牛熊发生的可靠概率。']
    (folder/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cards=[]
    for row in current['predictions']:
        h=row['horizon'];kind=row['candidate']
        description='规则输出，不提供概率估计。' if row['score_type']=='deterministic_rule' else '类别分数（未校准）：熊 {:.1%} · 震荡 {:.1%} · 牛 {:.1%}'.format(*row['scores'])
        cards.append(f'<article><h3>{HORIZONS[h]} · {h}交易日</h3><strong>{html.escape(row["label"])}</strong><p>{MODEL_NAMES[kind]}</p><small>{description}</small></article>')
    metric_rows=[]
    for h in s['protocol']['horizons']:
        chosen=s['choices'][str(h)]['candidate']
        for kind in s['protocol']['candidates']:
            m=s['test_scores'][str(h)][kind]
            metric_rows.append(f'<tr><td>{h}交易日</td><td>{MODEL_NAMES[kind]}{" · 验证选中" if kind==chosen else ""}</td><td>{m["n"]}</td><td>{m["accuracy"]:.1%}</td><td>{m["balanced_accuracy"]:.1%}</td><td>{m["macro_f1"]:.3f}</td></tr>')
    images=[]
    for name,caption in [('kline_10y.png','近十年月K线与成交量；红绿背景为事后20%反转确认的大波段，斜纹末段尚未确认。'),
                         ('history_cycles.png','完整历史收盘与事后波段。图示不等于当时可提前预测；确认日通常晚于拐点。')]:
        raw=base64.b64encode((output/name).read_bytes()).decode()
        images.append(f'<figure><img src="data:image/png;base64,{raw}" alt="上证指数历史趋势"><figcaption>{caption}</figcaption></figure>')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>上证指数牛熊趋势研究</title>
<style>body{{font:14px/1.8 "Microsoft YaHei",sans-serif;color:#222;background:white;margin:0}}main{{max-width:1240px;margin:auto;padding:28px;overflow-wrap:anywhere}}h1{{font-size:28px}}h2{{margin-top:32px}}p,small,figcaption{{color:#666}}.notice{{background:#f5f6f8;padding:14px;border-radius:6px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}article{{flex:1;min-width:240px;border:1px solid #ddd;padding:18px;border-radius:6px}}article strong{{font-size:21px}}figure{{margin:24px 0}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:620px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#f7f8fa}}small{{font-size:12px}}</style><main>
<h1>上证指数牛熊趋势研究</h1><p>独立项目 code_bear · 上证指数 sh000001 · 数据截至 {current['as_of']} · {len(frame):,}条日线</p>
<div class="notice">第一轮结果：模型尚未证明能稳定提前识别牛熊转折。下方历史波段是事后描述；未来趋势是按时间隔离回测选出的方案输出。多数类准确率较高时，也要看Macro-F1和平衡准确率。</div>
<h2>未来各期限趋势</h2><div class="cards">{''.join(cards)}</div>
<p>三个月/半年/一年分别使用±6%/±10%/±15%的累计收益阈值，并要求至少60%的未来收盘价在起点同方向一侧，其余记为震荡/混合。当前预测尚未走完期限，不能提前算作正确。</p>
<h2>历史大趋势与K线</h2>{''.join(images)}
<h2>2022年起逐年向前测试</h2><p>先用2010—2021年滚动验证选方案，再按年初可得标签重训测试。测试结果未用于选模型。类别分数未经概率校准；期限重叠，日样本并非独立牛熊周期。</p>
<div class="scroll"><table><thead><tr><th>期限</th><th>方案</th><th>样本数</th><th>准确率</th><th>平衡准确率</th><th>Macro-F1</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
<h2>数据与使用</h2><p>腾讯公开上证指数日线，原始响应、CSV、SQLite和来源清单均保存在本项目data目录。原始成交量单位未跨源校准，模型仅使用相对量；训练和模型在本项目独立保存。此页面内嵌图片，可单文件离线打开。</p>
<p>更新：运行collect.py后运行predict.py，再运行render.py生成新页面；重新研究训练使用train.py。旧训练报告与数据快照按版本保留。详细方法及逐条预测见reports/{s['run_id']}/。</p></main></html>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    print('REPORT='+str(folder.relative_to(ROOT)))


if __name__=='__main__':render()

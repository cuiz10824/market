"""Build a self-contained HTML snapshot, including a script-free table fallback."""
import html
import json
import re
import tempfile
from pathlib import Path


def atomic_text(path, text):
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix=path.stem+'-',suffix='.tmp',delete=False) as stream:
        temp=Path(stream.name)
        stream.write(text)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def table(payload,kind):
    names=['行业 / 指数','下一交易日' if kind=='forecast' else '当天','一周','一月','三月','六月']
    output=['<thead><tr>'+''.join('<th scope="col">'+n+'</th>' for n in names)+'</tr></thead><tbody>']
    for row in payload['rows']:
        name=row.get('display_name',row['name'])
        subtitle=(row['category'] if name==row['name'] else row['name'])+' · '+row['code']
        output.append('<tr><th scope="row"><span class="sector">'+html.escape(name)+
                      '</span><span class="sub">'+html.escape(subtitle)+
                      '</span><span class="sub">'+html.escape(row['as_of'])+'</span></th>')
        for h in payload['horizons']:
            value=row[kind][str(h)]
            color='neutral' if value is None or abs(value)<.005 else 'up' if value>0 else 'down'
            text='—' if value is None else f'{0 if abs(value)<.005 else value:+.2f}%' if value else '0.00%'
            if text=='+0.00%':text='0.00%'
            label=('<span class="range">'+html.escape(row['models'].get(str(h),'模型不可用'))+'</span>') if kind=='forecast' else ''
            output.append(f'<td class="{color}">{text}{label}</td>')
        output.append('</tr>')
    return ''.join(output)+'</tbody>'


def render_page(root,payload):
    folder=Path(root)/'html'
    template=(folder/'index.template.html').read_text(encoding='utf-8')
    style=(folder/'style.css').read_text(encoding='utf-8')
    template=template.replace('<link rel="stylesheet" href="style.css">','<style>\n'+style+'\n</style>')
    for kind in ('actual','forecast'):
        template=re.sub(r'(<table id="'+kind+r'">)(.*?)(</table>)',
                        lambda m:m[1]+m[2]+table(payload,kind)+m[3],template,flags=re.S)
    for name in ('model.js','snapshot.js','app.js'):
        if name=='snapshot.js':
            data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
            code='window.MARKET_SNAPSHOT = '+data+';'
        else:
            code=(folder/name).read_text(encoding='utf-8')
        code=re.sub('</script',r'<\\/script',code,flags=re.I)
        template=template.replace(f'<script src="{name}"></script>',f'<script>\n{code}\n</script>')
    dates=sorted({r['as_of'] for r in payload['rows']})
    date_text=dates[0]+(' ～ '+dates[-1] if len(dates)>1 else '')
    template=template.replace('数据截至 —','数据截至 '+html.escape(date_text))
    template=template.replace('正在载入本地预测…','已载入保存的预测快照')
    template=template.replace('>等待数据<','>已保存的本地快照<')
    template=template.replace('正在载入历史收盘数据与已训练的回归模型结果。联网更新请使用「启动网页.bat」。',
                              '当前显示已保存的真实收盘数据和模型预测。联网更新请使用「启动网页.bat」。')
    return template


def build_page(root,payload):
    atomic_text(Path(root)/'html/index.html',render_page(root,payload))

'use strict';
const M = MarketModel;
let data = Object.fromEntries(M.sectors.map(([name]) => [name, []]));
let mode = 'empty';
const $ = id => document.getElementById(id);
const labels = ['当天', '最近一周', '最近一月', '最近三月', '最近六月'];
const format = n => `${n > 0 ? '+' : ''}${n.toFixed(2)}%`;
function cell(value, range) {
  const el = document.createElement('td');
  if (value === null) {el.className='neutral'; el.textContent=mode==='empty'?'—':'数据不足'; el.style.fontSize='12px'; return el;}
  el.className = Math.abs(value)<.005 ? 'neutral' : value>0 ? 'up':'down';
  el.textContent = format(Math.abs(value)<.005?0:value);
  if (range) {const small=document.createElement('span'); small.className='range'; small.textContent=`${format(range.low)} ~ ${format(range.high)}`; el.append(small);}
  return el;
}
function render() {
  for (const kind of ['actual','forecast']) {
    const table=$(kind); table.replaceChildren();
    const head=table.createTHead().insertRow();
    ['行业 / 指数',...labels].forEach((label,i)=>{const th=document.createElement('th');th.scope='col';th.textContent=label; if(kind==='forecast'&&i===1) th.textContent='当天 · 下一交易日';head.append(th);});
    const body=table.createTBody();
    M.sectors.forEach(([name,code,desc])=>{
      const series=data[name];const tr=body.insertRow();const th=document.createElement('th');th.scope='row';
      const title=document.createElement('span');title.className='sector';title.textContent=name;
      const sub=document.createElement('span');sub.className='sub';sub.textContent=`${code} · ${desc}`;
      const date=document.createElement('span');date.className='sub';date.textContent=series.length?series.at(-1).date:'暂无数据';
      th.append(title,sub,date);tr.append(th);
      M.horizons.forEach(h=>{const p=kind==='forecast'?M.predict(series,h):null;tr.append(cell(kind==='actual'?M.actual(series,h):p?.value??null,p));});
    });
  }
  const dates=Object.values(data).filter(a=>a.length).map(a=>a.at(-1).date).sort();
  $('asof').textContent=dates.length?`数据截至 ${dates[0]}${dates[0]!==dates.at(-1)?' ~ '+dates.at(-1):''}`:'数据截至 —';
  $('source-badge').textContent={empty:'等待数据',live:'东方财富 · 历史收盘',import:'用户导入 · 未核验来源',demo:'演示数据 · 非真实行情'}[mode];
  $('export').disabled=mode==='empty';
}
function notice(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':'';}
$('demo').onclick=()=>{data=M.demo();mode='demo';render();$('status').textContent='演示模式';notice('当前上下两栏均使用程序生成的虚构行情，仅用于演示排版与计算。日期仅作示意，未匹配交易所节假日，不代表实际市场表现。');};
$('refresh').onclick=async()=>{
  if(location.protocol==='file:'){notice('在线行情需要本地服务：请先安装 Python 3，再双击「启动网页.bat」。当前也可导入 CSV 或查看演示。',true);return;}
  const button=$('refresh');button.disabled=true;$('status').textContent='正在获取七组指数历史行情…';
  try{
    const response=await fetch('/api/market',{signal:AbortSignal.timeout(65000)});
    const payload=await response.json(); if(!response.ok)throw Error(payload.error||'行情服务不可用');
    const next=M.validate(payload.rows);
    if(M.sectors.some(([name])=>next[name].length<127))throw Error('在线行情不完整，暂不替换已有数据');
    data=next;mode='live';render();$('status').textContent='行情已更新';
    const latest=Object.values(data).map(a=>a.at(-1).date).sort()[0];
    const age=(Date.now()-Date.parse(latest+'T16:00:00+08:00'))/86400000;
    notice(`已载入东方财富历史收盘数据。${age>4?'注意：行情距今超过四天，可能已过期，请核验来源。':'休市或未收盘时，展示最近已获得的交易日数据。'}预测仅为历史收益外推，详见下方方法说明。`);
  }catch(e){$('status').textContent='更新失败 · 保留原有数据';notice('无法获取行情：'+e.message+'。请检查网络或改用 CSV 导入；页面不会用演示数据代替真实行情。',true);}
  finally{button.disabled=false;}
};
$('import').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  try{if(file.size>5*1024*1024)throw Error('文件超过 5 MB');const next=M.parseCSV(await file.text());data=next;mode='import';render();$('status').textContent='已导入 CSV';notice('已载入本地 CSV，来源未独立核验。缺失行业或历史长度不足时显示“数据不足”；预测至少需要 21 条收盘记录。');}
  catch(err){notice('导入失败：'+err.message+'。原有数据已保留。',true);}finally{e.target.value='';}
};
$('export').onclick=()=>{
  const lines=['date,name,close'];for(const [name,series] of Object.entries(data))for(const p of series)lines.push(`${p.date},${name},${p.close}`);
  const url=URL.createObjectURL(new Blob(['\uFEFF'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=`A股-${mode==='demo'?'虚构演示':mode==='live'?'东方财富':'用户导入'}-历史收盘.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
render();
if(location.protocol!=='file:')$('refresh').click();

'use strict';
const M=MarketModel;
const $=id=>document.getElementById(id);
const labels=['当天','一周','一月','三月','六月'];
let payload=null;
let busy=false;
const offline=location.protocol==='file:';
const isStale=row=>row.stale||(Date.now()-Date.parse(row.as_of+'T16:00:00+08:00'))/86400000>4;
function notice(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':'';}
function render(next){
  payload=M.validate(next);
  for(const kind of ['actual','forecast']){
    const table=$(kind);table.replaceChildren();
    const head=table.createTHead().insertRow();
    ['行业 / 指数',...labels].forEach((name,i)=>{
      const th=document.createElement('th');th.scope='col';
      th.textContent=kind==='forecast'&&i===1?'下一交易日':name;head.append(th);
    });
    const body=table.createTBody();
    for(const row of payload.rows){
      const tr=body.insertRow();const th=document.createElement('th');th.scope='row';
      const name=row.display_name||row.name;
      for(const [className,text] of [['sector',name],['sub',`${name===row.name?row.category:row.name} · ${row.code}`],['sub',`${row.as_of}${isStale(row)?' · 日期较旧':''}`]]){
        const span=document.createElement('span');span.className=className;span.textContent=text;th.append(span);
      }
      th.title=row.selection_note||row.note;tr.append(th);
      for(const h of M.horizons){
        const td=tr.insertCell();const value=row[kind][h];
        td.className=value===null||Math.abs(value)<.005?'neutral':value>0?'up':'down';
        td.textContent=value===null?'—':M.format(value);
        if(kind==='forecast'){
          const span=document.createElement('span');span.className='range';
          span.textContent=row.models[h]||'模型不可用';td.append(span);
          td.title=`基于 ${row.as_of} 收盘；${h===1?'预测下一交易日收盘相对此次收盘的变化；':''}${row.models[h]}${value===0?'，不表示未来一定不涨不跌':''}`;
        }
      }
    }
  }
  const dates=[...new Set(payload.rows.map(r=>r.as_of))].sort();
  $('asof').textContent=`数据截至 ${dates[0]}${dates.length>1?' ～ '+dates.at(-1):''}`;
  $('source-badge').textContent=offline?'已保存的本地快照':'本地历史 + 行情增量';
  $('prediction-time').textContent=`预测生成于 ${new Date(payload.generated_at).toLocaleString('zh-CN')}`;
  const errors=Object.keys(payload.last_refresh.errors||{}).length;
  const zero=payload.rows.reduce((n,row)=>n+Object.values(row.models).filter(v=>v==='零变化基准').length,0);
  const details=[offline?'当前为保存的快照。双击「启动网页.bat」可更新行情并重新预测。':'启动时和每30分钟检查已收盘行情；页面每60秒读取最新结果。',
    `${payload.rows.length*M.horizons.length}个预测中有${zero}个使用零变化基准，0%不是确定不涨不跌。`,
    payload.enhancement?.status==='no_validation_upgrade'?`${payload.enhancement.lookback_sessions===1260?'五年历史特征':'本轮量价增强'}模型未通过验证替换门槛，继续展示旧模型结果。`:'',
    payload.enhancement?.upgraded_tasks>0?`${payload.enhancement.upgraded_tasks}项已启用通过验证的量价增强模型。`:'',
    errors?`${errors}个指数联网更新失败，保留已有数据与对应日期。`:'',
    payload.rows.some(r=>r.enhancement_error)?'部分量价特征未就绪，对应预测沿用旧模型。':'',
    payload.rows.some(isStale)?'部分行情距今超过4天，请核验数据日期。':''].filter(Boolean);
  notice(details.join(' '),errors>0);
  $('status').textContent=payload.refreshing?'正在检查新行情…':errors?'更新未全部成功 · 本地结果可用':'预测结果已载入';
  $('export').disabled=false;
}
async function load(){
  const response=await fetch('/api/dashboard',{cache:'no-store',signal:AbortSignal.timeout(15000)});
  if(!response.ok)throw Error('本地预测服务不可用');
  render(await response.json());
}
$('refresh').onclick=async()=>{
  if(offline){notice('请双击「启动网页.bat」启动每日预测。直接打开HTML时仅能查看保存的快照。',true);return;}
  if(busy)return;busy=true;$('refresh').disabled=true;
  try{
    const response=await fetch('/api/refresh',{method:'POST',signal:AbortSignal.timeout(10000)});
    if(!response.ok)throw Error('刷新请求失败');
    $('status').textContent='正在获取行情并计算预测…';
    for(let i=0;i<90;i++){
      await new Promise(resolve=>setTimeout(resolve,2000));await load();
      if(!payload.refreshing)break;
    }
  }catch(error){notice(error.message+'；保留当前显示的结果。',true);}
  finally{busy=false;$('refresh').disabled=false;}
};
$('export').onclick=()=>{
  const url=URL.createObjectURL(new Blob([M.toCSV(payload)],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='行业涨跌与回归预测.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
if(window.MARKET_SNAPSHOT){try{render(window.MARKET_SNAPSHOT);}catch(error){notice(error.message,true);}}
if(!offline){
  load().catch(error=>notice(error.message+'；当前显示已保存快照。',true));
  setInterval(()=>{if(!busy)load().catch(error=>notice(error.message+'；保留当前结果。',true));},60000);
}

/* Validate snapshots. Forecasts are computed only by the saved Python regressors. */
(function(root){
  'use strict';
  const horizons=[1,5,21,63,126];
  function validate(payload){
    if(payload?.schema_version!==1 || payload.unit!=='percent' || !Array.isArray(payload.rows) || payload.rows.length!==8) throw Error('预测数据格式不完整');
    if(!Array.isArray(payload.selected_codes) || payload.selected_codes.length!==8 || new Set(payload.selected_codes).size!==8) throw Error('指数配置无效');
    const codes=new Set();
    for(const row of payload.rows){
      if(!/^\d{6}$/.test(row.code) || !payload.selected_codes.includes(row.code) || codes.has(row.code) || !/^\d{4}-\d{2}-\d{2}$/.test(row.as_of)) throw Error('指数或日期无效');
      codes.add(row.code);
      for(const kind of ['actual','forecast']) for(const h of horizons){
        const value=row[kind]?.[h];
        if(value!==null && !Number.isFinite(value)) throw Error('涨跌幅数据无效');
      }
    }
    return payload;
  }
  function format(value){
    const magnitude=Math.abs(value);
    const text=magnitude>0&&magnitude<.0001?value.toExponential(2):value.toFixed(magnitude>0&&magnitude<.01?4:2);
    return `${value>0?'+':''}${text}%`;
  }
  function toCSV(payload){
    const lines=['code,name,as_of,kind,horizon_trading_days,return_percent,model,forecast_as_of'];
    for(const row of validate(payload).rows) for(const kind of ['actual','forecast']) for(const h of horizons){
      lines.push([row.code,row.name,row.as_of,kind,h,row[kind][h]??'',kind==='forecast'?(row.models[h]??'未训练'):'历史收盘',kind==='forecast'?(row.forecast_as_of?(row.forecast_as_of[h]??''):row.as_of):''].join(','));
    }
    return '\ufeff'+lines.join('\r\n');
  }
  const api={horizons,validate,format,toCSV};
  if(typeof module!=='undefined')module.exports=api;else root.MarketModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);

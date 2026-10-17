/* Shared pure calculations; no network or DOM. */
(function (root) {
  'use strict';
  const sectors = [
    ['沪深300', '000300', '沪深300'], ['能源', '000928', '中证能源'],
    ['医药', '000933', '中证医药'], ['消费', '000932', '中证消费'],
    ['制造业', '000930', '中证工业 · 行业近似'], ['金融', '000934', '中证金融'],
    ['科技', '000935', '中证信息 · 行业近似']
  ];
  const horizons = [1, 5, 21, 63, 126];
  function validate(rows) {
    if (!Array.isArray(rows) || !rows.length) throw Error('数据为空');
    const result = Object.fromEntries(sectors.map(([name]) => [name, []]));
    const seen = new Set();
    for (const r of rows) {
      const date = String(r.date || ''); const close = Number(r.close);
      if (!Object.hasOwn(result, r.name)) throw Error('未知行业：' + r.name);
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(date)) || new Date(date).toISOString().slice(0, 10) !== date) throw Error('无效日期：' + date);
      if (date > new Date().toLocaleDateString('sv-SE', {timeZone:'Asia/Shanghai'})) throw Error('不能导入未来日期：' + date);
      if (!Number.isFinite(close) || close <= 0) throw Error('收盘价必须为正数');
      const key = r.name + date;
      if (seen.has(key)) throw Error('行业日期重复：' + r.name + ' ' + date);
      seen.add(key); result[r.name].push({date, close});
    }
    Object.values(result).forEach(a => a.sort((a, b) => a.date.localeCompare(b.date)));
    return result;
  }
  function actual(series, h) {
    return series.length > h ? (series.at(-1).close / series.at(-1-h).close - 1) * 100 : null;
  }
  function predict(series, h) {
    if (series.length < 21) return null;
    const prices = series.slice(-253).map(p => p.close);
    const returns = prices.slice(1).map((v, i) => Math.log(v / prices[i]));
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + (b-mean)**2, 0) / (returns.length-1);
    const drift = mean * .25 * h;
    const spread = 1.2815515655 * Math.sqrt(variance * h);
    return {value: Math.expm1(drift)*100, low: Math.expm1(drift-spread)*100, high: Math.expm1(drift+spread)*100};
  }
  function parseCSV(text) {
    const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
    if (lines.shift()?.trim() !== 'date,name,close') throw Error('CSV 首行应为 date,name,close');
    const rows = lines.filter(l => l.trim()).map(line => {
      const parts = line.split(',').map(x => x.trim());
      if (parts.length !== 3) throw Error('CSV 每行必须为三列，不支持引号或列内逗号');
      return {date: parts[0], name: parts[1], close: parts[2]};
    });
    return validate(rows);
  }
  function demo() {
    const rows = []; let t = new Date(); t.setUTCHours(0,0,0,0); t.setUTCDate(t.getUTCDate()-1);
    const dates=[];
    while (dates.length < 280) { if (![0,6].includes(t.getUTCDay())) dates.unshift(t.toISOString().slice(0,10)); t.setUTCDate(t.getUTCDate()-1); }
    sectors.forEach(([name], s) => {let close = 1000; dates.forEach((date, i) => {close *= Math.exp(Math.sin(i*1.73+s*2.1)*.014 + Math.cos(i*.17+s)*.005 + (s-3)*.00015); rows.push({date,name,close});});});
    return validate(rows);
  }
  const api = {sectors,horizons,validate,actual,predict,parseCSV,demo};
  if (typeof module !== 'undefined') module.exports = api;
  else root.MarketModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);

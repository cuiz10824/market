const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const M=require('../model.js');
function snapshot(){
  const context={window:{}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../snapshot.js'),'utf8'),context);
  return JSON.parse(JSON.stringify(context.window.MARKET_SNAPSHOT));
}
test('snapshot covers 8 indices and trained regression horizons only',()=>{
  const p=M.validate(snapshot());
  assert.equal(p.rows.length,8);
  assert.equal(p.rows.flatMap(r=>Object.values(r.forecast)).filter(x=>x!==null).length,40);
  assert.ok(p.rows.every(r=>Number.isFinite(r.forecast[1]) && r.models[1]));
});
test('non-finite or incomplete predictions are rejected',()=>{
  const p=snapshot();p.rows[0].forecast[1]=Infinity;assert.throws(()=>M.validate(p));
  p.rows[0].forecast[1]=0;p.rows[0].forecast[21]=Infinity;assert.throws(()=>M.validate(p));
  p.rows.pop();assert.throws(()=>M.validate(p));
});
test('signed percentage formatting preserves small returns',()=>{
  assert.equal(M.format(.04948),'+0.05%');assert.equal(M.format(-1.23),'-1.23%');
  assert.equal(M.format(-.0001),'-0.0001%');assert.equal(M.format(0),'0.00%');
  assert.equal(M.format(.000001),'+1.00e-6%');
});
test('export contains 80 actual/forecast entries including trained single-day values',()=>{
  const csv=M.toCSV(snapshot());assert.equal(csv.split('\r\n').length,81);
  assert.match(csv,/000300,沪深300/);assert.match(csv,/forecast,1,-?[0-9.]+,LightGBM · 1年,2026-/);
  assert.doesNotMatch(csv,/零变化基准|历史平均收益/);
  assert.doesNotMatch(csv,/p_up|大涨|大跌/);
  assert.match(csv,/comparison_start,comparison_end,target_date_estimated/);
  const row=snapshot().rows[0];
  const range=row.date_ranges.actual['5'];
  assert.ok(csv.includes(`,${range.start},${range.end},false`));
});

const {test} = require('node:test');
const assert = require('node:assert/strict');
const M = require('../model.js');
test('all horizons compare the correct closing prices', () => {
  const s = Array.from({length: 140}, (_, i) => ({close: 100 + i}));
  for (const h of M.horizons) assert.equal(M.actual(s,h), (239/(239-h)-1)*100);
  assert.equal(M.actual(s.slice(0,126),126), null);
});
test('flat prices have zero forecast and interval', () => {
  const s = Array.from({length:30}, () => ({close:100}));
  assert.deepEqual(M.predict(s,126), {value:0,low:0,high:0});
  assert.equal(M.predict(s.slice(0,20),1),null);
});
test('log trend shrinks and uncertainty widens', () => {
  const s = Array.from({length:100}, (_,i)=>({close:100*Math.exp(.001*i + .02*Math.sin(i))}));
  const one=M.predict(s,1), half=M.predict(s,126);
  assert.ok(one.low<one.value && one.value<one.high);
  assert.ok(half.high-half.low > one.high-one.low);
});
test('CSV normalizes dates, sorts rows and rejects invalid values', () => {
  const parsed=M.parseCSV('\ufeffdate,name,close\r\n2025-01-03,能源,102\r\n2025-01-02,能源,100');
  assert.equal(M.actual(parsed.能源,1),2.0000000000000018);
  for(const row of ['2025-02-30,能源,100','2025-01-01,能源,-1','2025-01-01,未知,100','2099-01-01,能源,100'])
    assert.throws(()=>M.parseCSV('date,name,close\n'+row));
  assert.throws(()=>M.parseCSV('date,name,close\n2025-01-01,能源,100\n2025-01-01,能源,101'));
});
test('demo covers all sectors and every horizon without non-finite predictions', () => {
  const data=M.demo();
  for (const [name] of M.sectors) for (const h of M.horizons) {
    assert.ok(Number.isFinite(M.actual(data[name],h)));
    assert.ok(Object.values(M.predict(data[name],h)).every(Number.isFinite));
  }
});

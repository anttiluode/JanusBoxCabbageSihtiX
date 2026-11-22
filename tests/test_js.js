// node tests/test_js.js  -- the browser core must agree with the Python core.
const J = require('../web/jbx.js'), fx = require('./js_fixture.json');
const S = J.S, x = new Float64Array(S * S);
let mean = 0;
for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) {
  let v = 0.2 * Math.sin(0.05 * i + 0.03 * j);
  if ((i - 120) ** 2 + (j - 90) ** 2 < 2500) v += 1;
  if (Math.abs(i - 180) < 30 && Math.abs(j - 170) < 45) v -= 0.7;
  if (j - 0.6 * i > 120) v += 0.5;
  x[i * S + j] = v; mean += v;
}
mean /= S * S; for (let n = 0; n < S * S; n++) x[n] -= mean;
const { y, td } = J.degrade(x), { F, Y } = J.features(y);
let worst = 0, fails = 0;
const chk = (name, a, b) => { const e = Math.abs(a - b) / (Math.abs(b) + 1e-6); worst = Math.max(worst, e); if (e > 1e-6) { fails++; console.log('MISMATCH', name, a, b); } };
fx.pix.forEach(([a, b], k) => {
  const n = a * J.M + b;
  chk('y', y[n], fx.y[k]); chk('td', td[n], fx.td[k]);
  for (const nm in fx.features) chk(nm, F[nm][n], fx.features[nm][k]);
});
const pd = J.writeDark(F, fx.w), pe = J.writeEQ(Y, fx.weq);
fx.pix.forEach(([a, b], k) => { const n = a * J.M + b; chk('dark', pd[n], fx.dark_pred[k]); chk('eq', pe[n], fx.eq_pred[k]); });
chk('r2', J.r2(pd, td), fx.r2_dark);
console.log(fails ? `FAIL ${fails}` : `ok  worst rel err ${worst.toExponential(2)}  r2_dark ${fx.r2_dark.toFixed(4)}`);
process.exit(fails ? 1 : 0);

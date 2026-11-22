// JanusBox core in plain JS: same bands, features and writers as jbx/*.py.
// Works in the browser (window.JBX) and in node (module.exports).
(function (root) {
  const S = 256, M = 512, K = 4, MARGIN = 24, N = M * M;
  const f = i => (i < M / 2 ? i : i - M) / M;
  const R = new Float64Array(N), TH = new Float64Array(N);
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
    const fy = f(i), fx = f(j); R[i * M + j] = Math.hypot(fx, fy); TH[i * M + j] = Math.atan2(fy, fx);
  }
  const rise = (r, lo, hi) => {
    const t = Math.min(1, Math.max(0, (Math.log2(Math.max(r, 1e-12)) - Math.log2(lo)) / (Math.log2(hi) - Math.log2(lo))));
    return Math.sin(t * Math.PI / 2);
  };
  const fact = n => (n <= 1 ? 1 : n * fact(n - 1));
  const alpha = Math.sqrt(2 ** (2 * (K - 1)) * fact(K - 1) ** 2 / (K * fact(2 * (K - 1))));
  const RF = new Float64Array(N), LOW2 = new Float64Array(N), RP = new Float64Array(N), RG = new Float64Array(N);
  const DARKF = new Float64Array(N), TRANS = new Float64Array(N), RF2 = new Float64Array(N);
  const ANG = [...Array(K)].map(() => new Float64Array(N));
  for (let n = 0; n < N; n++) {
    const r = R[n], rf = rise(r, 0.125, 0.25), rp1 = rise(r, 0.0625, 0.125);
    RF[n] = rf; RF2[n] = rf * rf; LOW2[n] = 1 - rf * rf;
    RP[n] = Math.sqrt(1 - rf * rf) * rp1;
    RG[n] = Math.sqrt(1 - rp1 * rp1) * rise(r, 0.03125, 0.0625);
    const dark = r >= 0.25 ? 1 : 0; DARKF[n] = rf * rf * dark; TRANS[n] = 1 - dark;
    for (let k = 0; k < K; k++) { const c = Math.cos(TH[n] - Math.PI * k / K); ANG[k][n] = c > 0 ? 2 * alpha * Math.abs(c) ** (K - 1) : 0; }
  }
  const KNOTS = [...Array(9)].map((_, j) => 0.125 + j * 0.125 / 8);
  const EQB = KNOTS.map(c => { const h = KNOTS[1] - KNOTS[0], a = new Float64Array(N);
    for (let n = 0; n < N; n++) a[n] = Math.max(0, 1 - Math.abs(R[n] - c) / h) * RF2[n] * TRANS[n]; return a; });

  // radix-2 FFT on rows then columns, in place on (re, im)
  const rev = new Uint32Array(M); { const bits = Math.log2(M); for (let i = 0; i < M; i++) { let x = i, y = 0; for (let b = 0; b < bits; b++) { y = (y << 1) | (x & 1); x >>= 1; } rev[i] = y; } }
  const cosT = new Float64Array(M / 2), sinT = new Float64Array(M / 2);
  for (let i = 0; i < M / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / M); sinT[i] = Math.sin(2 * Math.PI * i / M); }
  const br = new Float64Array(M), bi = new Float64Array(M);
  function fft1(re, im, off, stride, inv) {
    for (let i = 0; i < M; i++) { br[rev[i]] = re[off + i * stride]; bi[rev[i]] = im[off + i * stride]; }
    for (let size = 2; size <= M; size <<= 1) {
      const half = size >> 1, step = M / size;
      for (let st = 0; st < M; st += size) for (let j = 0; j < half; j++) {
        const c = cosT[j * step], s = inv ? sinT[j * step] : -sinT[j * step];
        const a = st + j, b = a + half, tr = br[b] * c - bi[b] * s, ti = br[b] * s + bi[b] * c;
        br[b] = br[a] - tr; bi[b] = bi[a] - ti; br[a] += tr; bi[a] += ti;
      }
    }
    const sc = inv ? 1 / M : 1;
    for (let i = 0; i < M; i++) { re[off + i * stride] = br[i] * sc; im[off + i * stride] = bi[i] * sc; }
  }
  function fft2(re, im, inv) { for (let i = 0; i < M; i++) fft1(re, im, i * M, 1, inv); for (let j = 0; j < M; j++) fft1(re, im, j, M, inv); }
  function FFT(x) { const re = Float64Array.from(x), im = new Float64Array(N); fft2(re, im, false); return [re, im]; }
  function filt([Yr, Yi], H) {          // complex result of ifft(Y * H)
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let n = 0; n < N; n++) { re[n] = Yr[n] * H[n]; im[n] = Yi[n] * H[n]; }
    fft2(re, im, true); return [re, im];
  }
  const realFilt = (x, H) => filt(FFT(x), H)[0];

  function extend(x) {                 // x: Float64Array S*S -> M*M mirror
    const e = new Float64Array(N);
    for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
      const ii = i < S ? i : M - 1 - i, jj = j < S ? j : M - 1 - j; e[i * M + j] = x[ii * S + jj];
    }
    return e;
  }
  function crop(e) { const x = new Float64Array(S * S); for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) x[i * S + j] = e[i * M + j]; return x; }
  function roll(e, dy, dx) { const o = new Float64Array(N); for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) o[((i + dy) % M) * M + ((j + dx) % M)] = e[i * M + j]; return o; }

  function degrade(x) {
    const X = FFT(extend(x));
    return { y: filt(X, LOW2)[0], t: filt(X, RF2)[0], td: filt(X, DARKF)[0] };
  }

  // Features, named exactly as in jbx/writers.py
  function features(y) {
    const Y = FFT(y);
    const P = ANG.map(A => { const H = new Float64Array(N); for (let n = 0; n < N; n++) H[n] = RP[n] * A[n]; return filt(Y, H); });
    const Q = ANG.map(A => { const H = new Float64Array(N); for (let n = 0; n < N; n++) H[n] = RG[n] * A[n]; return filt(Y, H); });
    let ms = 0; for (const [pr] of P) for (let n = 0; n < N; n++) ms += pr[n] * pr[n];
    const s = Math.sqrt(ms / (K * N)) + 1e-9, F = {}, pcs = [];
    for (let k = 0; k < K; k++) {
      const [pr, pi] = P[k], [qr, qi] = Q[k];
      const m = {}; for (const nm of ['clip', 'pow2', 'tri', 'triI', 'absb', 'dbl', 'dblI']) { m[nm] = new Float64Array(N); m[nm + '*'] = new Float64Array(N); }
      const pc4 = new Float64Array(N);
      for (let n = 0; n < N; n++) {
        const b = pr[n], c = pi[n], ab = Math.hypot(b, c), a = ab + 1e-9 * s;
        const pc = Math.hypot(b + qr[n], c + qi[n]) / (ab + Math.hypot(qr[n], qi[n]) + 1e-6 * s), g = pc ** 4;
        pc4[n] = g;
        const r2 = b * b - c * c, i2 = 2 * b * c, r3 = r2 * b - i2 * c, i3 = r2 * c + i2 * b;
        const v = { clip: Math.max(-s, Math.min(s, b)), pow2: Math.sign(b) * b * b / s, tri: r3 / (a * a), triI: i3 / (a * a),
                    absb: Math.abs(b), dbl: r2 / a, dblI: i2 / a };
        for (const nm in v) { m[nm][n] = v[nm]; m[nm + '*'][n] = v[nm] * g; }
      }
      for (const nm of ['clip', 'pow2', 'tri', 'triI', 'absb', 'dbl', 'dblI']) { F[nm + k] = m[nm]; F[nm + k + '*pc'] = m[nm + '*']; }
      pcs.push(pc4);
    }
    return { F, P, pcs, s, Y };
  }
  function writeDark(F, w, transform) {
    const acc = new Float64Array(N);
    for (const nm in w) { const v = transform ? transform(F[nm]) : F[nm], c = w[nm]; for (let n = 0; n < N; n++) acc[n] += c * v[n]; }
    return realFilt(acc, DARKF);
  }
  function writeEQ(Y, weq) {
    const H = new Float64Array(N); EQB.forEach((B, j) => { for (let n = 0; n < N; n++) H[n] += weq[j] * B[n]; });
    return filt(Y, H)[0];
  }
  function r2(pred, truth) {             // on interior of the crop
    let e = 0, t = 0;
    for (let i = MARGIN; i < S - MARGIN; i++) for (let j = MARGIN; j < S - MARGIN; j++) {
      const n = i * M + j, d = truth[n] - (pred ? pred[n] : 0); e += d * d; t += truth[n] * truth[n];
    }
    return 1 - e / t;
  }
  function phaseTwin(x, seed) {
    let st = seed >>> 0; const rnd = () => { st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
    const nre = new Float64Array(S * S).map(() => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()));
    // S-sized FFTs via embedding is awkward; do a direct S x S DFT with a small radix-2 FFT.
    const fftS = (re, im, inv) => {
      const L = S, rv = new Uint32Array(L), bits = Math.log2(L);
      for (let i = 0; i < L; i++) { let a = i, b = 0; for (let q = 0; q < bits; q++) { b = (b << 1) | (a & 1); a >>= 1; } rv[i] = b; }
      const tr = new Float64Array(L), ti = new Float64Array(L);
      const one = (off, stride) => {
        for (let i = 0; i < L; i++) { tr[rv[i]] = re[off + i * stride]; ti[rv[i]] = im[off + i * stride]; }
        for (let size = 2; size <= L; size <<= 1) { const h = size >> 1;
          for (let s0 = 0; s0 < L; s0 += size) for (let j = 0; j < h; j++) {
            const ang = (inv ? 2 : -2) * Math.PI * j / size, c = Math.cos(ang), s = Math.sin(ang), a = s0 + j, b = a + h;
            const xr = tr[b] * c - ti[b] * s, xi = tr[b] * s + ti[b] * c; tr[b] = tr[a] - xr; ti[b] = ti[a] - xi; tr[a] += xr; ti[a] += xi; } }
        for (let i = 0; i < L; i++) { re[off + i * stride] = tr[i] / (inv ? L : 1); im[off + i * stride] = ti[i] / (inv ? L : 1); }
      };
      for (let i = 0; i < L; i++) one(i * L, 1); for (let j = 0; j < L; j++) one(j, L);
    };
    const xr = Float64Array.from(x), xi = new Float64Array(S * S); fftS(xr, xi, false);
    const zi = new Float64Array(S * S); fftS(nre, zi, false);
    for (let n = 0; n < S * S; n++) { const mag = Math.hypot(xr[n], xi[n]), ph = Math.atan2(zi[n], nre[n]); xr[n] = mag * Math.cos(ph); xi[n] = mag * Math.sin(ph); }
    fftS(xr, xi, true); return xr;
  }
  function edgeScene(seed) {
    let st = (seed * 2654435761) >>> 0; const u = (a, b) => { st = (st * 1664525 + 1013904223) >>> 0; return a + (b - a) * st / 4294967296; };
    const ss = 4, NN = S * ss, img = new Float64Array(NN * NN);
    const gx = u(-1, 1), gy = u(-1, 1);
    for (let i = 0; i < NN; i++) for (let j = 0; j < NN; j++) img[i * NN + j] = 0.3 * (gx * j / NN + gy * i / NN);
    const nsh = 6 + Math.floor(u(0, 7));
    for (let s = 0; s < nsh; s++) {
      const cx = u(.05, .95), cy = u(.05, .95), val = u(-1, 1), sx = u(-1, 1), sy = u(-1, 1), kind = Math.floor(u(0, 3)), a = u(0, Math.PI);
      const p1 = kind === 0 ? u(.04, .25) : kind === 1 ? u(.03, .2) : u(.06, .22), p2 = kind === 2 ? 0 : (kind === 0 ? u(.04, .25) : u(.03, .2));
      const ca = Math.cos(a), sa = Math.sin(a);
      for (let i = 0; i < NN; i++) for (let j = 0; j < NN; j++) {
        const xx = j / NN - cx, yy = i / NN - cy, uu = xx * ca + yy * sa, vv = -xx * sa + yy * ca;
        const inside = kind === 0 ? (uu / p1) ** 2 + (vv / p2) ** 2 < 1 : kind === 1 ? Math.abs(uu) < p1 && Math.abs(vv) < p2 : vv > -p1 / 2 && vv < p1 - 1.7 * Math.abs(uu);
        if (inside) img[i * NN + j] = val + 0.25 * (sx * xx + sy * yy);
      }
    }
    const x = new Float64Array(S * S); let mean = 0;
    for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) { let acc = 0; for (let a = 0; a < ss; a++) for (let b = 0; b < ss; b++) acc += img[(i * ss + a) * NN + j * ss + b]; x[i * S + j] = acc / 16; mean += x[i * S + j]; }
    mean /= S * S; for (let n = 0; n < S * S; n++) x[n] -= mean; return x;
  }
  const api = { S, M, K, MARGIN, N, FFT, filt, realFilt, extend, crop, roll, degrade, features, writeDark, writeEQ, r2, phaseTwin, edgeScene, DARKF, RF2 };
  if (typeof module !== 'undefined') module.exports = api; else root.JBX = api;
})(this);

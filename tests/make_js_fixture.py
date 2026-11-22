"""Reference numbers for tests/test_js.js (python tests/make_js_fixture.py)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jbx import bands as B
from jbx.writers import raw_features, predict_dark, predict_eq


def scene():
    i, j = np.mgrid[0:B.S, 0:B.S].astype(float)
    x = 0.2 * np.sin(0.05 * i + 0.03 * j)
    x += 1.0 * (((i - 120) ** 2 + (j - 90) ** 2) < 50 ** 2)
    x -= 0.7 * ((np.abs(i - 180) < 30) & (np.abs(j - 170) < 45))
    x += 0.5 * ((j - 0.6 * i) > 120)
    return x - x.mean()


if __name__ == '__main__':
    x = scene(); y, t, td = B.degrade(x); F = raw_features(y)
    pix = [(30, 40), (100, 130), (125, 140), (180, 125), (200, 215), (60, 200), (240, 10), (300, 400)]
    names = ['clip0', 'pow21*pc', 'tri2', 'triI3*pc', 'absb0*pc', 'dbl1', 'dblI2', 'tri0*pc']
    w = {'tri0': 0.6, 'triI2*pc': -0.4, 'clip1': 0.2}
    weq = list(np.linspace(0.5, 1.5, 9))
    pd = predict_dark(F, list(w), list(w.values())); pe = predict_eq(y, weq)
    d = lambda a: B.interior(B.crop(a))
    r2 = lambda p, q: float(1 - np.sum((d(q) - d(p)) ** 2) / np.sum(d(q) ** 2))
    out = {'pix': pix, 'y': [float(y[a, b]) for a, b in pix], 'td': [float(td[a, b]) for a, b in pix],
           'features': {n: [float(F[n][a, b]) for a, b in pix] for n in names},
           'w': w, 'weq': weq, 'dark_pred': [float(pd[a, b]) for a, b in pix], 'eq_pred': [float(pe[a, b]) for a, b in pix],
           'r2_dark': r2(pd, td)}
    Path(__file__).with_name('js_fixture.json').write_text(json.dumps(out, indent=1))
    print('wrote', out['r2_dark'])

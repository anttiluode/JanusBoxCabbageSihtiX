"""results/figure.png: what each writer puts in the dark band, on held-out images."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jbx import bands as B
from jbx.data import photo_crops, edge_world, phase_twin, PHOTO_TEST
from jbx.writers import raw_features, predict_dark, predict_eq

ROOT = Path(__file__).parent


def r2(p, t):
    a, b = B.interior(B.crop(p)), B.interior(B.crop(t))
    return 1 - np.sum((b - a) ** 2) / np.sum(b ** 2)


def main():
    W = json.loads((ROOT / 'results/weights.json').read_text())
    rows = [('edge scene (held out), weights: edge', edge_world(2, 1)[0][1], 'edge'),
            ('same scene, phase twin, weights: edge', phase_twin(edge_world(2, 1)[0][1], 2000), 'edge'),
            ('astronaut crop (held out), weights: photo', photo_crops(['astronaut.png'])[1][1], 'photo')]
    cols = ['blurred input', 'truth', 'true dark band', 'even translator', 'odd, no flashlight', 'flashlight + odd']
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols), 2.5 * len(rows)))
    for r, (label, x, world) in enumerate(rows):
        y, t, td = B.degrade(x); F = raw_features(y)
        preds = {}
        for name, key in [('even translator', 'even (sign-blind translator)'),
                          ('odd, no flashlight', 'odd translator, no flashlight'),
                          ('flashlight + odd', 'flashlight + odd translator')]:
            w = W[world][key]; preds[name] = predict_dark(F, list(w), list(w.values()))
        lo, hi = np.percentile(x, [0.5, 99.5]); m = np.percentile(np.abs(B.crop(td)), 99.5)
        for c, name in enumerate(cols):
            a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
            if name == 'blurred input': a.imshow(B.crop(y), cmap='gray', vmin=lo, vmax=hi)
            elif name == 'truth': a.imshow(x, cmap='gray', vmin=lo, vmax=hi)
            elif name == 'true dark band': a.imshow(B.crop(td), cmap='gray', vmin=-m, vmax=m)
            else:
                a.imshow(B.crop(preds[name]), cmap='gray', vmin=-m, vmax=m)
                a.set_xlabel(f'dark R² {r2(preds[name], td):+.3f}', fontsize=9)
            if r == 0: a.set_title(name, fontsize=10)
        ax[r, 0].set_ylabel(label, fontsize=8)
    fig.suptitle('Dark band (r ≥ 0.25 cyc/px): the blurred input has zero energy here, so any Fourier gain writes 0 (R² = 0)', fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / 'results/figure.png', dpi=110)
    print('results/figure.png')


if __name__ == '__main__':
    main()

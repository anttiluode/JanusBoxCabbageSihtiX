"""Does letting the flashlight choose the translator by local texture rescue photographs?

    python moe_experiment.py      # ~45 min on 2 CPU cores -> results/moe_receipt.json, results/moe_weights.json

Every configuration is scored on the same held-out images and controls as
experiment.py, plus three new ones:

  gate from another image   trained AND tested with gates read from a different image:
                            same number of weights, arbitrary partition. If texture
                            gating only adds capacity, this matches it.
  gate shifted              trained texture weights, gate maps rolled 64x96 px at test
  within-image ceiling      fit on 2/3 of a test image's own pixels, score the other 1/3:
                            the most a readout of this form could get on that image
"""
import json, os, time
from pathlib import Path
import numpy as np
from jbx import bands as B
from jbx.data import PHOTO_TRAIN, PHOTO_TEST, photo_crops, edge_world, phase_twin
from jbx.writers import WRITERS, raw_features, Gram
from jbx.moe import TextureGate, moe_columns, post_eq_columns, POST_PER_FEATURE, DESCRIPTORS

OUT = Path(__file__).parent / 'results'
SEL = WRITERS['flashlight + odd translator']
ROLL = lambda v: np.roll(v, (64, 96), axis=(0, 1))
CONFIGS = [('photo', 1, 'texture'), ('photo', 3, 'texture'), ('photo', 6, 'texture'), ('photo', 12, 'texture'),
           ('photo', 6, 'other image'), ('photo', 1, 'post-EQ'),
           ('mixed', 1, 'texture'), ('mixed', 6, 'texture'), ('mixed', 1, 'post-EQ')]


def ncols(c):
    return len(SEL) * (POST_PER_FEATURE if c[2] == 'post-EQ' else c[1])


def columns(c, F, g, transform=None, gate_transform=None):
    if c[2] == 'post-EQ':
        return post_eq_columns(F, SEL, transform)
    return moe_columns(F, SEL, g, transform, gate_transform)


def key(c):
    return f'{c[0]} | G={c[1]} | {c[2]}'


def main():
    t0 = time.time()
    photo_tr, edge_tr = photo_crops(PHOTO_TRAIN), edge_world(1, 60)
    photo_te, edge_te = photo_crops(PHOTO_TEST), edge_world(2, 20)
    if os.environ.get('SMOKE'):
        photo_tr, edge_tr, photo_te, edge_te = photo_tr[:4], edge_tr[:3], photo_te[:3], edge_te[:2]
        CONFIGS[:] = [('photo', 1, 'texture'), ('photo', 3, 'texture'), ('photo', 3, 'other image'), ('photo', 1, 'post-EQ'), ('mixed', 3, 'texture')]
        globals()['OUT'] = Path(os.environ['SMOKE'])
    sets = {'photo test': photo_te, 'edge test': edge_te,
            'photo twins': [(n + ' twin', phase_twin(x, 1000 + i)) for i, (n, x) in enumerate(photo_te)],
            'edge twins': [(n + ' twin', phase_twin(x, 2000 + i)) for i, (n, x) in enumerate(edge_te)]}
    Y = lambda x: B.degrade(x)[0]

    # Gates are fitted on training inputs only.
    ys_photo = [Y(x) for _, x in photo_tr]
    ys_edge = [Y(x) for _, x in edge_tr]
    gates = {}
    for c in CONFIGS:
        w, G, _ = c
        if (w, G) not in gates:
            gates[(w, G)] = None if G == 1 else TextureGate(G, seed=0).fit(ys_photo + (ys_edge if w == 'mixed' else []))
    print('gates fitted', round(time.time() - t0), 's', flush=True)

    def gate_maps(c, y, other_y):
        tg = gates[(c[0], c[1])]
        if tg is None:
            return np.ones((1, B.M, B.M))
        return tg.gates(other_y if c[2] == 'other image' else y)

    n = len(SEL)
    train = {key(c): Gram(ncols(c)) for c in CONFIGS}
    for name_list, ys, worlds in [(photo_tr, ys_photo, ('photo', 'mixed')), (edge_tr, ys_edge, ('mixed',))]:
        for i, (nm, x) in enumerate(name_list):
            y = ys[i]; other = ys[(i + 7) % len(ys)]
            F = raw_features(y)
            b = B.interior(B.crop(B.degrade(x)[2])).ravel()
            for c in CONFIGS:
                if c[0] in worlds:
                    train[key(c)].add(columns(c, F, gate_maps(c, y, other)), b)
        print('train', worlds, round(time.time() - t0), 's', flush=True)
    weights = {k: g.solve() for k, g in train.items()}

    receipt = {'descriptors': DESCRIPTORS, 'configs': {}, 'per_image': {}, 'ceiling': {}}
    for c in CONFIGS:
        k = key(c)
        receipt['configs'][k] = {'n_weights': ncols(c), 'train_r2': float(train[k].r2(weights[k]))}
        if gates[(c[0], c[1])] is not None:
            receipt['configs'][k]['experts'] = gates[(c[0], c[1])].centres()

    for sname, items in sets.items():
        ys = [Y(x) for _, x in items]
        acc = {key(c): {'clean': Gram(ncols(c))} for c in CONFIGS}
        controls = sname in ('photo test', 'edge test')
        for i, (nm, x) in enumerate(items):
            y = ys[i]; nxt = ys[(i + 1) % len(ys)]
            b = B.interior(B.crop(B.degrade(x)[2])).ravel()
            F = raw_features(y)
            Fw = raw_features(nxt) if controls else None
            hold = np.arange(len(b)) % 3 == 0
            if sname == 'photo test':
                # Wide within-image ceiling: every odd/gated feature at 9 offsets (+-3 px), 288 weights.
                from jbx.bands import apply, crop as _crop, RF as _RF, DARK as _DK
                fm = [_crop(apply(F[nn], _RF ** 2 * _DK)) for nn in SEL]
                Aw = np.column_stack([np.roll(f, (dy, dx), axis=(0, 1))[24:B.S - 24, 24:B.S - 24].ravel()
                                      for f in fm for dy in (-3, 0, 3) for dx in (-3, 0, 3)])
                gf = Gram(Aw.shape[1]); gf.add(Aw[~hold], b[~hold])
                gh = Gram(Aw.shape[1]); gh.add(Aw[hold], b[hold])
                receipt['ceiling'].setdefault('wide: 32 features x 9 offsets, fitted per image', []).append([nm, float(gh.r2(gf.solve()))])
            for c in CONFIGS:
                k = key(c); w = weights[k]
                other = Y(items[(i + 7) % len(items)][1]) if c[2] == 'other image' else None
                g = gate_maps(c, y, other)
                A = columns(c, F, g)
                acc[k]['clean'].add(A, b)
                r = 1 - np.sum((b - A @ w) ** 2) / np.sum(b ** 2)
                receipt['per_image'].setdefault(sname, {}).setdefault(k, []).append([nm, float(r)])
                if sname == 'photo test':
                    gf = Gram(A.shape[1]); gf.add(A[~hold], b[~hold])
                    gh = Gram(A.shape[1]); gh.add(A[hold], b[hold])
                    receipt['ceiling'].setdefault(k, []).append([nm, float(gh.r2(gf.solve()))])
                if controls:
                    for cn, cols in [('parent shifted 64x96', lambda: columns(c, F, g, ROLL, ROLL)),
                                     ('parent from next image', lambda: columns(
                                         c, Fw, g if c[1] == 1 or c[2] == 'other image' else gates[(c[0], c[1])].gates(nxt))),
                                     ('gate shifted 64x96', lambda: columns(c, F, g, None, ROLL))]:
                        if cn == 'gate shifted 64x96' and c[1] == 1:
                            continue
                        acc[k].setdefault(cn, Gram(ncols(c))).add(cols(), b)
        for c in CONFIGS:
            k = key(c)
            for cn, gram in acc[k].items():
                receipt['configs'][k][sname if cn == 'clean' else f'{sname}: {cn}'] = float(gram.r2(weights[k]))
        print('scored', sname, round(time.time() - t0), 's', flush=True)

    for k, lst in receipt['per_image']['photo test'].items():
        v = np.array([r for _, r in lst])
        receipt['configs'][k]['photo crops positive'] = f'{int((v > 0).sum())}/{len(v)}'
        receipt['configs'][k]['photo crop median'] = float(np.median(v))
    for k, lst in receipt['ceiling'].items():
        med = float(np.median([r for _, r in lst]))
        if k in receipt['configs']:
            receipt['configs'][k]['within-image ceiling (median photo crop)'] = med
        else:
            receipt['wide_ceiling_median'] = med
    receipt['seconds'] = round(time.time() - t0)
    OUT.mkdir(exist_ok=True)
    (OUT / 'moe_receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    (OUT / 'moe_weights.json').write_text(json.dumps({k: w.tolist() for k, w in weights.items()}) + '\n')
    for k, v in receipt['configs'].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'experts'})


if __name__ == '__main__':
    main()

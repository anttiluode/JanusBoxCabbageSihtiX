"""Train every writer, score it on held-out images, run the controls and the outside reader.

    python experiment.py            # ~10-20 min on a CPU, writes results/receipt.json + results/weights.json
"""
import json, time
from pathlib import Path
import numpy as np
from jbx import bands as B
from jbx.data import PHOTO_TRAIN, PHOTO_TEST, photo_crops, edge_world, phase_twin
from jbx.writers import (ALL, EVEN, ODD, WRITERS, raw_features, dark_filtered, eq_columns, Gram,
                         predict_dark, predict_eq, EQ_KNOTS)
from jbx.reader import probe, energy_frac_dark

OUT = Path(__file__).parent / 'results'
IDX = {n: i for i, n in enumerate(ALL)}


class Stats:
    """Gram of the full dictionary against one target; any writer is a sub-block."""
    def __init__(self):
        self.g = Gram(len(ALL)); self.per = []
    def add(self, A, b, name):
        self.g.add(A, b)
        self.per.append((name, A.T @ A, A.T @ b, float(b @ b)))


def sub(stats_g, sel):
    ix = [IDX[n] for n in sel]
    g = Gram(len(ix)); g.AtA = stats_g.AtA[np.ix_(ix, ix)]; g.Atb = stats_g.Atb[ix]; g.btb = stats_g.btb
    return g


def r2_from(AtA, Atb, btb, w, sign=1.0):
    return float(1 - (btb - 2 * sign * w @ Atb + w @ AtA @ w) / btb)


def pass_images(items, eq=None, roll=None, wrong=None):
    """Accumulate dark-band Grams (and transition Grams for the EQ)."""
    dark, trans = Stats(), Gram(len(EQ_KNOTS))
    ys = []
    for i, (name, x) in enumerate(items):
        y, t, td = B.degrade(x)
        ys.append(y)
        src = y if wrong is None else B.degrade(items[(i + 1) % len(items)][1])[0]
        F = raw_features(src)
        tf = (lambda v: np.roll(v, roll, axis=(0, 1))) if roll else None
        A = dark_filtered(F, ALL, tf)
        dark.add(A, B.interior(B.crop(td)).ravel(), name)
        tt = B.interior(B.crop(B.apply(B.extend(x), B.RF ** 2 * B.TRANS))).ravel()
        trans.add(eq_columns(y), tt)
    return dark, trans, ys


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    worlds = {
        'photo': (photo_crops(PHOTO_TRAIN), photo_crops(PHOTO_TEST)),
        'edge': (edge_world(1, 60), edge_world(2, 20)),
    }
    worlds['mixed'] = (worlds['photo'][0] + worlds['edge'][0], None)
    tests = {'photo test': worlds['photo'][1], 'edge test': worlds['edge'][1]}
    tests['photo twins'] = [(n + ' twin', phase_twin(x, 1000 + i)) for i, (n, x) in enumerate(tests['photo test'])]
    tests['edge twins'] = [(n + ' twin', phase_twin(x, 2000 + i)) for i, (n, x) in enumerate(tests['edge test'])]
    print({k: len(v[0]) for k, v in worlds.items()}, {k: len(v) for k, v in tests.items()}, flush=True)

    train_stats = {}
    for w in ('photo', 'edge'):
        train_stats[w] = pass_images(worlds[w][0])[:2]
        print('trained pass', w, round(time.time() - t0), 's', flush=True)
    mixed_dark = Gram(len(ALL)); mixed_trans = Gram(len(EQ_KNOTS))
    for w in ('photo', 'edge'):
        for G, S_ in ((mixed_dark, train_stats[w][0].g), (mixed_trans, train_stats[w][1])):
            G.AtA += S_.AtA; G.Atb += S_.Atb; G.btb += S_.btb
    train_grams = {'photo': train_stats['photo'][0].g, 'edge': train_stats['edge'][0].g, 'mixed': mixed_dark}
    eq_grams = {'photo': train_stats['photo'][1], 'edge': train_stats['edge'][1], 'mixed': mixed_trans}

    test_stats = {}
    for tn, items in tests.items():
        test_stats[tn] = pass_images(items)
        print('test pass', tn, round(time.time() - t0), 's', flush=True)
    controls = {
        'rolled parent (64,96 px)': pass_images(tests['photo test'], roll=(64, 96))[0],
        'wrong parent (next image)': pass_images(tests['photo test'], wrong=True)[0],
        'edge: rolled parent': pass_images(tests['edge test'], roll=(64, 96))[0],
        'edge: wrong parent': pass_images(tests['edge test'], wrong=True)[0],
    }
    print('controls', round(time.time() - t0), 's', flush=True)

    receipt = {'bands': {'dark_zone': 'r >= 0.25 cycles/px', 'parent_centre': 0.125, 'K': B.K,
                         'size': B.S, 'margin': B.MARGIN},
               'counts': {'train': {k: len(v[0]) for k, v in worlds.items()},
                          'test': {k: len(v) for k, v in tests.items()}},
               'input_dark_energy_max': max(energy_frac_dark(y) for st in test_stats.values() for y in st[2]),
               'magnitude_only_dark_r2': 0.0,
               'writers': {}, 'eq': {}, 'controls': {}, 'flip': {}}
    weights = {}
    for tw, G in train_grams.items():
        for wn, sel in WRITERS.items():
            g = sub(G, sel); w = g.solve()
            row = {'train_r2': float(g.r2(w)), 'n_weights': len(sel)}
            for tn, st in test_stats.items():
                gt = sub(st[0].g, sel); row[tn] = float(gt.r2(w))
            receipt['writers'].setdefault(tw, {})[wn] = row
            weights.setdefault(tw, {})[wn] = dict(zip(sel, w.tolist()))
        weq = eq_grams[tw].solve()
        receipt['eq'][tw] = {'weights': weq.tolist(), 'knots': EQ_KNOTS.tolist(),
                             **{tn: float(st[1].r2(weq)) for tn, st in test_stats.items()}}
        weights.setdefault(tw, {})['eq'] = weq.tolist()

    main_w = 'flashlight + odd translator'
    for tw in ('photo', 'edge', 'mixed'):
        sel = WRITERS[main_w]; w = np.array([weights[tw][main_w][n] for n in sel])
        for cn, st in controls.items():
            receipt['controls'].setdefault(tw, {})[cn] = float(sub(st.g, sel).r2(w))
        for wn, sel2 in WRITERS.items():
            w2 = np.array([weights[tw][wn][n] for n in sel2])
            gt = sub(test_stats['photo test'][0].g, sel2)
            ge = sub(test_stats['edge test'][0].g, sel2)
            # Negating y flips odd features, leaves even ones, flips the target:
            # equivalent to scoring the same target with even weights negated.
            wneg = np.array([-v if n.split('*')[0].rstrip('0123456789') in EVEN else v
                             for n, v in zip(sel2, w2)])
            receipt['flip'].setdefault(tw, {})[wn] = {
                'photo: same image': float(gt.r2(w2)), 'photo: negative image': float(gt.r2(wneg)),
                'edge: same image': float(ge.r2(w2)), 'edge: negative image': float(ge.r2(wneg))}
    # Whole fine band: transition (EQ) + dark (translator), per test world.
    full = {}
    for tw in ('photo', 'edge', 'mixed'):
        sel = WRITERS[main_w]; w = np.array([weights[tw][main_w][n] for n in sel])
        weq = np.array(weights[tw]['eq'])
        for tn in ('photo test', 'edge test'):
            dg = sub(test_stats[tn][0].g, sel); tg = test_stats[tn][1]
            e_d, e_t = dg.btb, tg.btb
            sse_d = e_d * (1 - dg.r2(w)); sse_t = e_t * (1 - tg.r2(weq))
            full.setdefault(tw, {})[tn] = {
                'blur (nothing added)': 0.0,
                'EQ on what survived (magnitude-only)': float(1 - (sse_t + e_d) / (e_t + e_d)),
                'EQ + flashlight/odd translator': float(1 - (sse_t + sse_d) / (e_t + e_d)),
                'dark share of fine-band energy': float(e_d / (e_t + e_d))}
    receipt['full_fine_band'] = full
    # Per-image dark R2 for the mixed main writer.
    sel = WRITERS[main_w]; w = np.array([weights['mixed'][main_w][n] for n in sel]); ix = [IDX[n] for n in sel]
    receipt['per_image_mixed_main'] = {tn: [(nm, r2_from(AtA[np.ix_(ix, ix)], Atb[ix], btb, w))
                                            for nm, AtA, Atb, btb in st[0].per] for tn, st in test_stats.items()}

    # Outside reader on five boxes. It sees inputs and outputs only.
    probe_ys = test_stats['photo test'][2][:3] + test_stats['edge test'][2][:3]
    def box_from(sel, w, w_eq=None):
        def box(y):
            F = raw_features(y)
            out = y + predict_dark(F, sel, w)
            return out if w_eq is None else out + predict_eq(y, w_eq)
        return box
    wm = weights['mixed']
    boxes = {
        'trained flashlight + odd': (box_from(WRITERS[main_w], [wm[main_w][n] for n in WRITERS[main_w]],
                                              np.array(wm['eq'])), wm[main_w]),
        'trained even (sign-blind)': (box_from(WRITERS['even (sign-blind translator)'],
                                               [wm['even (sign-blind translator)'][n] for n in WRITERS['even (sign-blind translator)']]),
                                      wm['even (sign-blind translator)']),
        'hidden tripler (0.6 tri0, -0.4 triI2)': (box_from(['tri0', 'triI2'], [0.6, -0.4]), {'tri0': 0.6, 'triI2': -0.4}),
        'magnitude-only EQ': (lambda y: y + predict_eq(y, np.array(wm['eq'])), {}),
    }
    rng = np.random.default_rng(7)
    rphase = np.exp(2j * np.pi * rng.random((B.M, B.M)))
    def random_phase_box(y):
        P = B.analytic(y, B.RP)
        return y + B.apply(sum(np.real(np.abs(p) * rphase) for p in P), B.RF ** 2 * B.DARK)
    boxes['flashlight with random phase'] = (random_phase_box, None)
    receipt['reader'] = {}
    for bn, (box, truth) in boxes.items():
        rep = probe(box, probe_ys)
        if truth:
            keys = set(truth) | set(rep.get('recovered', {}))
            tv = np.array([truth.get(k, 0.0) for k in keys]); rv = np.array([rep.get('recovered', {}).get(k, 0.0) for k in keys])
            rep['weight_error_rel'] = float(np.linalg.norm(tv - rv) / max(np.linalg.norm(tv), 1e-30))
        receipt['reader'][bn] = rep
        print('reader', bn, rep['rung'], round(time.time() - t0), 's', flush=True)

    receipt['seconds'] = round(time.time() - t0)
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (OUT / 'weights.json').write_text(json.dumps(weights, indent=1) + '\n')
    print(json.dumps({k: receipt[k] for k in ('writers', 'controls', 'full_fine_band')}, indent=1))


if __name__ == '__main__':
    main()

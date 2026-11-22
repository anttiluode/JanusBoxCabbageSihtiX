"""The outside reader: which rung is this box on, and what is it writing?

The reader may send any input it likes into a box and receives the box's
output image. It never sees the box's weights. Four probes:

1. Dark energy: does the output contain frequencies above 0.25 cycles/pixel,
   where the input has none? An EQ (any fixed or input-written Fourier gain)
   cannot put energy there.
2. Additivity: is box(y1 + y2) = box(y1) + box(y2) in the dark band?
3. Parity: send the negative image. Odd writers flip, even writers do not.
4. Dictionary: residue-guided greedy selection (Luotain2 style) over the
   shared feature dictionary, scored on held-out pixels. It reports which
   features it needed, their weights, and how much of the dark band is left
   unexplained.
"""
import numpy as np
from .bands import DARK, RADIUS, apply, crop, interior
from .writers import ALL, raw_features, dark_filtered


def dark_part(ze):
    return interior(crop(apply(ze, DARK)))


def energy_frac_dark(ze):
    Z = np.abs(np.fft.fft2(ze)) ** 2
    return float(Z[RADIUS >= 0.25].sum() / max(Z.sum(), 1e-30))


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def probe(box, ys, holdout_every=3, max_terms=40, min_gain=1e-3):
    """box: extended input -> extended output. ys: list of extended blurred inputs."""
    report = {}
    outs = [box(y) for y in ys]
    report['input_dark_energy_frac'] = max(energy_frac_dark(y) for y in ys)
    report['output_dark_energy_frac'] = float(np.mean([energy_frac_dark(z) for z in outs]))
    darks = [dark_part(z) for z in outs]
    total = sum(float(np.sum(d ** 2)) for d in darks)
    report['creates_new_frequencies'] = bool(total > 1e-12 * sum(float(np.sum(interior(crop(y)) ** 2)) for y in ys))
    if not report['creates_new_frequencies']:
        report['rung'] = 'EQ / linear filter: nothing written where the input is dark'
        return report
    add = dark_part(box(ys[0] + ys[1]))
    report['additivity_error'] = rel(add, darks[0] + darks[1])
    neg = [dark_part(box(-y)) for y in ys[:3]]
    odd_err = np.mean([rel(n, -d) for n, d in zip(neg, darks[:3])])
    even_err = np.mean([rel(n, d) for n, d in zip(neg, darks[:3])])
    report['parity'] = 'odd' if odd_err < 1e-6 else ('even' if even_err < 1e-6 else 'mixed')
    report['odd_error'] = float(odd_err); report['even_error'] = float(even_err)
    # Dictionary recovery on interleaved pixel sets.
    cols, tgt = [], []
    for y, d in zip(ys, darks):
        cols.append(dark_filtered(raw_features(y), ALL)); tgt.append(d.ravel())
    A = np.vstack(cols); b = np.concatenate(tgt)
    idx = np.arange(len(b)); hold = idx % holdout_every == 0; fit = ~hold
    Gf, cf = A[fit].T @ A[fit], A[fit].T @ b[fit]
    Gh, ch, bh = A[hold].T @ A[hold], A[hold].T @ b[hold], float(b[hold] @ b[hold])
    def heldout(sel):
        w = np.linalg.lstsq(Gf[np.ix_(sel, sel)], cf[sel], rcond=None)[0]
        return (bh - 2 * w @ ch[sel] + w @ Gh[np.ix_(sel, sel)] @ w) / bh, w
    chosen, history, cur = [], [], 1.0
    for _ in range(max_terms):
        cands = [(heldout(chosen + [j])[0], j) for j in range(A.shape[1]) if j not in chosen]
        res, j = min(cands)
        if cur - res < min_gain * cur:
            break
        chosen.append(j); cur = float(res)
        history.append({'added': ALL[j], 'heldout_residue': cur})
        if cur < 1e-12:
            break
    G, c = A.T @ A, A.T @ b
    w = np.linalg.lstsq(G[np.ix_(chosen, chosen)], c[chosen], rcond=None)[0] if chosen else np.zeros(0)
    report['recovered'] = {ALL[j]: float(wj) for j, wj in zip(chosen, w)}
    report['history'] = history
    report['unexplained_fraction'] = float(cur)
    if cur > 0.5:
        report['rung'] = 'writes new frequencies, but outside the dictionary (climb higher)'
    else:
        report['rung'] = f"bend in the analytic band ({report['parity']} translator)"
    return report

"""Texture-gated mixture of translators.

The flashlight no longer only says *where*; it also says *which* translator.
Five local texture descriptors are read from the blurred input (all degree 0,
so contrast cannot move a pixel between experts):

  agree   amplitude-weighted cross-scale phase agreement   sum|p|pc / sum|p|
  orient  orientation concentration                        max_k E_k / sum_k E_k
  slope   parent / grandparent energy, log                 log(E_p / E_q)
  level   local parent energy against the image's own, log log(E_p / s^2)
  peaky   4th/2nd moment of |p| (edge vs texture)          <sum|p|^4> / <sum|p|^2>^2

(all local averages are Gaussian, sigma = SIGMA px). k-means on training
pixels gives G texture centres; each pixel gets soft gates g_j (softmax of
-BETA * squared standardised distance). The writer is

  t_dark_hat = D( sum_j g_j(x) sum_m w_jm F_m(x) )

still linear in the weights, so it is fitted in closed form. With G = 1 it is
exactly the single flashlight + odd translator writer.
"""
import numpy as np
from .bands import M, S, K, RP, RG, RADIUS, RF, DARK, analytic, crop, interior
from .writers import EPS

SIGMA = 8.0
BETA = 1.0
DESCRIPTORS = ['agree', 'orient', 'slope', 'level', 'peaky']
_G2 = np.exp(-2 * (np.pi * SIGMA * RADIUS) ** 2)          # Gaussian blur in frequency
_HD = (RF ** 2 * DARK)[:, :M // 2 + 1]                     # dark filter, rfft layout


def smooth(v):
    """Gaussian local average of a non-negative map (clipped: FFT round-off can dip below 0)."""
    return np.maximum(np.real(np.fft.ifft2(np.fft.fft2(v) * _G2)), 0)


def descriptors(ye):
    """(5, M, M) degree-0 texture maps of an extended blurred input."""
    P = analytic(ye, RP)
    Q = analytic(ye, RG)
    s2 = np.mean([np.mean(np.real(p) ** 2) for p in P]) + EPS
    tiny = 1e-6 * s2
    Ek = [smooth(np.abs(p) ** 2) for p in P]
    Ep = sum(Ek) + tiny
    Eq = smooth(sum(np.abs(q) ** 2 for q in Q)) + tiny
    pcw = sum(smooth(np.abs(p) * np.abs(p + q) / (np.abs(p) + np.abs(q) + 1e-3 * np.sqrt(s2))) for p, q in zip(P, Q))
    aw = sum(smooth(np.abs(p)) for p in P) + 1e-3 * np.sqrt(s2)
    m4 = smooth(sum(np.abs(p) ** 4 for p in P)) + tiny ** 2
    return np.stack([pcw / aw,
                     np.max(Ek, axis=0) / Ep,
                     np.log(Ep / Eq),
                     np.log(Ep / s2),
                     np.log(m4 / Ep ** 2 + 1e-12)])


class TextureGate:
    def __init__(self, G, seed=0):
        self.G, self.seed = G, seed

    def fit(self, ys, per_image=4000):
        rng = np.random.default_rng(self.seed)
        X = []
        for y in ys:
            d = descriptors(y)[:, :S, :S].reshape(5, -1).T
            X.append(d[rng.choice(len(d), per_image, replace=False)])
        X = np.vstack(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        # k-means++ then Lloyd
        C = [Z[rng.integers(len(Z))]]
        for _ in range(1, self.G):
            d2 = np.min([np.sum((Z - c) ** 2, 1) for c in C], axis=0)
            C.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
        C = np.array(C)
        for _ in range(100):
            lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), 1)
            newC = np.array([Z[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(self.G)])
            if np.allclose(newC, C):
                break
            C = newC
        self.C = C
        self.share = np.bincount(lab, minlength=self.G) / len(lab)
        return self

    def gates(self, ye):
        Z = (np.moveaxis(descriptors(ye), 0, -1) - self.mu) / self.sd
        logit = -BETA * ((Z[..., None, :] - self.C) ** 2).sum(-1)
        logit -= logit.max(-1, keepdims=True)
        g = np.exp(logit)
        return np.moveaxis(g / g.sum(-1, keepdims=True), -1, 0)     # (G, M, M)

    def centres(self):
        return {f'expert {j}': dict(zip(DESCRIPTORS, (self.C[j] * self.sd + self.mu).round(3).tolist()),
                                    pixel_share=round(float(self.share[j]), 3)) for j in range(self.G)}


def dark_columns(maps):
    """Stack of extended maps (n, M, M) -> dark-filtered interior columns (pixels, n)."""
    out = []
    for i in range(0, len(maps), 16):
        Z = np.fft.irfft2(np.fft.rfft2(maps[i:i + 16]) * _HD, s=(M, M))
        out.append(Z[:, 24:S - 24, 24:S - 24].reshape(len(Z), -1))
    return np.vstack(out).T


def moe_columns(F, sel, gates, transform=None, gate_transform=None):
    """Dark-filtered interior columns of g_j * F_m, expert-major (column j*len(sel)+m)."""
    Fs = np.stack([F[n] if transform is None else transform(F[n]) for n in sel])
    cols = []
    for g in gates:
        g = g if gate_transform is None else gate_transform(g)
        cols.append(dark_columns(g[None] * Fs))
    return np.hstack(cols)


def moe_predict(F, sel, gates, w):
    """Full extended dark-band prediction."""
    acc = np.tensordot(w.reshape(len(gates), len(sel)), np.stack([F[n] for n in sel]), axes=([1], [0]))
    acc = np.sum(gates * acc, 0)
    return np.fft.irfft2(np.fft.rfft2(acc) * _HD, s=(M, M))


# ---- bend, then EQ: each translator feature gets its own dark-zone filter ----
# Radial tents across the dark zone, each in two phases: even (zero-phase) and
# quadrature along the feature's own orientation (a directional Hilbert pair).
POST_KNOTS = np.array([0.25, 1 / 3, 5 / 12, 0.5])


def _post_bank():
    from .bands import THETA
    h = POST_KNOTS[1] - POST_KNOTS[0]
    r = np.minimum(RADIUS, POST_KNOTS[-1])
    tents = [np.maximum(0, 1 - np.abs(r - c) / h) * RF ** 2 * DARK for c in POST_KNOTS]
    bank = {}
    for k in range(K):
        q = 1j * np.sign(np.cos(THETA - np.pi * k / K))
        bank[k] = [t[:, :M // 2 + 1] for t in tents] + [(t * q)[:, :M // 2 + 1] for t in tents]
    return bank


POST_BANK = _post_bank()
POST_PER_FEATURE = 2 * len(POST_KNOTS)


def post_eq_columns(F, sel, transform=None):
    """Columns (pixels, len(sel) * 8): feature-major, then filter."""
    cols = []
    for n in sel:
        v = F[n] if transform is None else transform(F[n])
        k = int(n.split('*')[0][-1])
        V = np.fft.rfft2(v)
        Z = np.fft.irfft2(np.stack([V * H for H in POST_BANK[k]]), s=(M, M))
        cols.append(Z[:, 24:S - 24, 24:S - 24].reshape(len(Z), -1))
    return np.vstack(cols).T

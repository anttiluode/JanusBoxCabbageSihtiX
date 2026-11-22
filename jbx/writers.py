"""Writers that regrow the dark band from the blurred input, and their dictionary.

Every writer is a weighted sum of feature maps computed from the blurred
input only, then passed through the fixed dark-band filter:

    t_dark_hat = D( sum_m w_m F_m(y) ),     D = fine band x dark zone

so the weights are found by plain least squares and each one can be read.

Feature families, per orientation k (p = parent analytic coefficient,
q = grandparent, b = Re p, s = one global scale per image):

  flashlight  pc  = |p + q| / (|p| + |q|)      unsigned cross-scale agreement in [0, 1]
  odd         clip(b, -s, s), sign(b) b^2 / s, Re/Im p^3/|p|^2  (flip the image -> flips)
  even        |b|, Re/Im p^2/|p|                                (flip the image -> unchanged)
  gated       any of the above times pc^4

All features are homogeneous of degree 1 in y, so every writer ignores
contrast. Odd features make the writer flip with the image; even ones cannot.
"""
import numpy as np
from .bands import K, RF, DARK, TRANS, RP, RG, RADIUS, analytic, apply, crop, extend, interior

ODD = ['clip', 'pow2', 'tri', 'triI']
EVEN = ['absb', 'dbl', 'dblI']
EPS = 1e-9


def raw_features(ye):
    """Dictionary of extended feature maps for an extended blurred input."""
    P = analytic(ye, RP)
    Q = analytic(ye, RG)
    s = np.sqrt(np.mean([np.mean(np.real(p) ** 2) for p in P])) + EPS
    F = {}
    for k in range(K):
        p, q = P[k], Q[k]
        b = np.real(p)
        a = np.abs(p) + EPS * s
        pc = np.abs(p + q) / (np.abs(p) + np.abs(q) + 1e-6 * s)
        base = {
            'clip': np.clip(b, -s, s),
            'pow2': np.sign(b) * b * b / s,
            'tri': np.real(p ** 3 / a ** 2),
            'triI': np.imag(p ** 3 / a ** 2),
            'absb': np.abs(b),
            'dbl': np.real(p ** 2 / a),
            'dblI': np.imag(p ** 2 / a),
        }
        g = pc ** 4
        for n, v in base.items():
            F[f'{n}{k}'] = v
            F[f'{n}{k}*pc'] = v * g
    return F


def names(kinds, gated):
    out = []
    for k in range(K):
        for n in kinds:
            if gated in ('plain', 'both'):
                out.append(f'{n}{k}')
            if gated in ('gated', 'both'):
                out.append(f'{n}{k}*pc')
    return out


ALL = names(ODD + EVEN, 'both')

WRITERS = {
    'even (sign-blind translator)': names(EVEN, 'both'),
    'odd translator, no flashlight': names(ODD, 'plain'),
    'flashlight + odd translator': names(ODD, 'both'),
    'tripled clock hands only': names(['tri', 'triI'], 'both'),
}

EQ_KNOTS = np.linspace(0.125, 0.25, 9)


def eq_basis():
    """Tent functions in radius over the transition zone: a radial EQ on what survived the blur."""
    out = []
    for j, c in enumerate(EQ_KNOTS):
        h = EQ_KNOTS[1] - EQ_KNOTS[0]
        out.append(np.maximum(0, 1 - np.abs(RADIUS - c) / h) * RF ** 2 * TRANS)
    return out


EQ_BASIS = eq_basis()


def dark_filtered(F, sel, transform=None):
    """Feature maps -> dark-band-filtered interior vectors (columns)."""
    cols = []
    for n in sel:
        v = F[n] if transform is None else transform(F[n])
        cols.append(interior(crop(apply(v, RF ** 2 * DARK))).ravel())
    return np.column_stack(cols)


def eq_columns(ye):
    Y = np.fft.fft2(ye)
    return np.column_stack([interior(crop(np.real(np.fft.ifft2(Y * H)))).ravel() for H in EQ_BASIS])


class Gram:
    """Accumulate normal equations so many images fit in memory."""
    def __init__(self, n):
        self.AtA = np.zeros((n, n)); self.Atb = np.zeros(n); self.btb = 0.0
    def add(self, A, b):
        self.AtA += A.T @ A; self.Atb += A.T @ b; self.btb += b @ b
    def solve(self, ridge=1e-9):
        n = len(self.Atb)
        lam = ridge * np.trace(self.AtA) / max(n, 1)
        return np.linalg.solve(self.AtA + lam * np.eye(n), self.Atb)
    def r2(self, w):
        sse = self.btb - 2 * w @ self.Atb + w @ self.AtA @ w
        return 1 - sse / self.btb


def predict_dark(F, sel, w):
    """Full extended dark-band prediction (one filter pass, linear in features)."""
    acc = sum(wi * F[n] for n, wi in zip(sel, w))
    return apply(acc, RF ** 2 * DARK)


def predict_eq(ye, w_eq):
    Y = np.fft.fft2(ye)
    H = sum(wi * Hi for wi, Hi in zip(w_eq, EQ_BASIS))
    return np.real(np.fft.ifft2(Y * H))

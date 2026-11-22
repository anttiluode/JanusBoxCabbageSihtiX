"""Fixed frequency geometry shared by every writer and by the reader.

A 256x256 image is mirror-extended to 512x512 so that its FFT is periodic
without a seam. Frequencies are in cycles per pixel (axis Nyquist = 0.5).

    fine band   R_f : 0 below 0.125, 1 above 0.25   (what the blur removed)
    low pass    L   : sqrt(1 - R_f^2)               (what the blur kept)
    parent      R_p : the octave below, centred near 0.125
    grandparent R_g : the octave below that, centred near 0.0625
    dark zone       : r >= 0.25. The blurred input has exactly zero energy here.

Oriented analytic bands use K = 4 steerable angular windows restricted to one
half-plane, so a parent coefficient p_k(x) is complex: |p_k| is the local
amplitude (the flashlight) and arg p_k is the local phase (the clock hand).
"""
from math import factorial
import numpy as np

S = 256          # working image size
M = 2 * S        # mirror-extended FFT size
K = 4            # orientations
MARGIN = 24      # interior pixels used for every score

_f = np.fft.fftfreq(M)
FY, FX = np.meshgrid(_f, _f, indexing="ij")
RADIUS = np.hypot(FX, FY)
THETA = np.arctan2(FY, FX)


def rise(r, lo, hi):
    """0 below lo, 1 above hi, raised cosine in log2 radius."""
    t = np.clip((np.log2(np.maximum(r, 1e-12)) - np.log2(lo)) / (np.log2(hi) - np.log2(lo)), 0, 1)
    return np.sin(t * np.pi / 2)


RF = rise(RADIUS, 0.125, 0.25)
LOW = np.sqrt(1 - RF ** 2)
RP = LOW * rise(RADIUS, 0.0625, 0.125)
RG = np.sqrt(1 - rise(RADIUS, 0.0625, 0.125) ** 2) * rise(RADIUS, 0.03125, 0.0625)
DARK = (RADIUS >= 0.25).astype(float)
TRANS = 1.0 - DARK

_alpha = np.sqrt(2 ** (2 * (K - 1)) * factorial(K - 1) ** 2 / (K * factorial(2 * (K - 1))))
ANGULAR = []
for _k in range(K):
    _c = np.cos(THETA - np.pi * _k / K)
    ANGULAR.append(2 * _alpha * np.abs(_c) ** (K - 1) * (_c > 0))   # one-sided: analytic


def extend(x):
    """Mirror-extend an SxS image to MxM (periodic, seam-free)."""
    return np.block([[x, x[:, ::-1]], [x[::-1], x[::-1, ::-1]]])


def crop(xe):
    return xe[:S, :S]


def interior(x):
    return x[MARGIN:S - MARGIN, MARGIN:S - MARGIN]


def apply(xe, H):
    return np.real(np.fft.ifft2(np.fft.fft2(xe) * H))


def degrade(x):
    """Return (extended input y, extended truth fine band t, extended truth dark band)."""
    X = np.fft.fft2(extend(x))
    y = np.real(np.fft.ifft2(X * LOW ** 2))
    t = np.real(np.fft.ifft2(X * RF ** 2))
    td = np.real(np.fft.ifft2(X * RF ** 2 * DARK))
    return y, t, td


def analytic(ye, radial):
    """Four complex oriented coefficient maps of an extended image."""
    Y = np.fft.fft2(ye)
    return [np.fft.ifft2(Y * radial * A) for A in ANGULAR]

"""Three image worlds: real photographs, an edge world, and phase-scrambled twins."""
import os
import numpy as np
from .bands import S

try:
    import skimage.data as _skd
    from skimage import io as _io, color as _color
    _SKDIR = os.path.dirname(_skd.__file__)
except Exception:   # pragma: no cover
    _SKDIR = None

PHOTO_TRAIN = ['camera.png', 'coins.png', 'moon.png', 'text.png', 'page.png', 'brick.png',
               'gravel.png', 'grass.png', 'rocket.jpg', 'horse.png', 'retina.jpg',
               'motorcycle_left.png', 'clock_motion.png', 'cell.png', 'hubble_deep_field.jpg', 'ihc.png']
PHOTO_TEST = ['astronaut.png', 'coffee.png', 'chelsea.png', 'grace_hopper.jpg']


def _path(name):
    if name == 'grace_hopper.jpg':
        import matplotlib
        return os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'sample_data', name)
    return os.path.join(_SKDIR, name)


def gray(name):
    im = _io.imread(_path(name))
    if im.ndim == 3:
        im = _color.rgb2gray(im[..., :3])
    im = im.astype(float)
    return im / (255.0 if im.max() > 1.5 else 1.0)


def photo_crops(names):
    out = []
    for n in names:
        im = gray(n)
        H, W = im.shape
        for i in range(0, H - S + 1, S):
            for j in range(0, W - S + 1, S):
                c = im[i:i + S, j:j + S]
                out.append((n, c - c.mean()))
    return out


def edge_scene(rng, n_shapes=None, ss=4):
    """Shaded shapes rendered at ss x supersampling then box-averaged: anti-aliased edges."""
    N = S * ss
    yy, xx = np.mgrid[0:N, 0:N] / N
    img = 0.3 * (rng.uniform(-1, 1) * xx + rng.uniform(-1, 1) * yy)
    for _ in range(n_shapes or rng.integers(6, 13)):
        cx, cy = rng.uniform(0.05, 0.95, 2)
        val = rng.uniform(-1, 1)
        shade = 0.25 * (rng.uniform(-1, 1) * (xx - cx) + rng.uniform(-1, 1) * (yy - cy))
        kind = rng.integers(3)
        a = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
        v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
        if kind == 0:
            mask = (u / rng.uniform(.04, .25)) ** 2 + (v / rng.uniform(.04, .25)) ** 2 < 1
        elif kind == 1:
            mask = (np.abs(u) < rng.uniform(.03, .2)) & (np.abs(v) < rng.uniform(.03, .2))
        else:
            r = rng.uniform(.06, .22)
            mask = (v > -r / 2) & (v < r - 1.7 * np.abs(u))
        img = np.where(mask, val + shade, img)
    img = img.reshape(S, ss, S, ss).mean(axis=(1, 3))
    return img - img.mean()


def edge_world(seed, count):
    rng = np.random.default_rng(seed)
    return [(f'edge{seed}_{i}', edge_scene(rng)) for i in range(count)]


def phase_twin(x, seed):
    """Same Fourier magnitudes, uniformly random (Hermitian) phases."""
    rng = np.random.default_rng(seed)
    X = np.fft.fft2(x)
    ph = np.angle(np.fft.fft2(rng.standard_normal(x.shape)))
    t = np.real(np.fft.ifft2(np.abs(X) * np.exp(1j * ph)))
    return t - t.mean()

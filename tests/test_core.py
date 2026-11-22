"""Fast checks of the claims the receipt rests on.  python -m pytest -q"""
import numpy as np
import pytest
from jbx import bands as B
from jbx.data import edge_world, phase_twin
from jbx.writers import raw_features, predict_dark, predict_eq, ALL, EVEN, EQ_KNOTS
from jbx.reader import probe, energy_frac_dark


@pytest.fixture(scope='module')
def scenes():
    return [x for _, x in edge_world(99, 3)]


@pytest.fixture(scope='module')
def ys(scenes):
    return [B.degrade(x)[0] for x in scenes]


def test_blur_leaves_dark_zone_empty(ys):
    assert max(energy_frac_dark(y) for y in ys) < 1e-24


def test_degrade_splits_the_image(scenes):
    x = scenes[0]
    y, t, _ = B.degrade(x)
    assert np.allclose(B.crop(y + t), x, atol=1e-10)


def test_any_fourier_gain_writes_nothing_dark(ys):
    rng = np.random.default_rng(0)
    for y in ys:
        out = predict_eq(y, rng.normal(size=len(EQ_KNOTS)) * 5)
        assert energy_frac_dark(out) < 1e-24
        H = np.exp(rng.normal(size=(B.M, B.M)))           # any gain at all
        assert energy_frac_dark(B.apply(y, H)) < 1e-24


def test_features_are_degree_one(ys):
    F1, F2 = raw_features(ys[0]), raw_features(3.7 * ys[0])
    for n in ALL:
        assert np.allclose(F2[n], 3.7 * F1[n], rtol=1e-6, atol=1e-9 * np.abs(F1[n]).max()), n


def test_parity_odd_flip_even_stay(ys):
    F, G = raw_features(ys[0]), raw_features(-ys[0])
    for n in ALL:
        even = n.split('*')[0].rstrip('0123456789') in EVEN
        ref = F[n] if even else -F[n]
        assert np.allclose(G[n], ref, atol=1e-9 * np.abs(F[n]).max()), n


def test_phase_twin_keeps_magnitudes(scenes):
    x = scenes[0]
    tw = phase_twin(x, 5)
    assert np.allclose(np.abs(np.fft.fft2(tw)), np.abs(np.fft.fft2(x)), rtol=1e-8, atol=1e-8)
    assert np.corrcoef(tw.ravel(), x.ravel())[0, 1] < 0.2


def test_reader_names_the_eq_rung(ys):
    w = np.linspace(1, 2, len(EQ_KNOTS))
    rep = probe(lambda y: y + predict_eq(y, w), ys)
    assert not rep['creates_new_frequencies'] and rep['rung'].startswith('EQ')


def test_reader_recovers_hidden_tripler(ys):
    sel, w = ['tri0', 'triI2'], [0.6, -0.4]
    rep = probe(lambda y: y + predict_dark(raw_features(y), sel, w), ys)
    assert rep['parity'] == 'odd'
    assert set(rep['recovered']) == set(sel)
    for n, v in zip(sel, w):
        assert abs(rep['recovered'][n] - v) < 1e-6
    assert rep['unexplained_fraction'] < 1e-10


def test_reader_flags_even_box(ys):
    sel = ['absb1*pc', 'dbl3']
    rep = probe(lambda y: y + predict_dark(raw_features(y), sel, [0.5, 0.3]), ys)
    assert rep['parity'] == 'even'
    assert set(rep['recovered']) == set(sel)


def test_reader_flags_out_of_dictionary(ys):
    rng = np.random.default_rng(7)
    ph = np.exp(2j * np.pi * rng.random((B.M, B.M)))
    def box(y):
        P = B.analytic(y, B.RP)
        return y + B.apply(sum(np.real(np.abs(p) * ph) for p in P), B.RF ** 2 * B.DARK)
    rep = probe(box, ys)
    assert rep['creates_new_frequencies'] and rep['unexplained_fraction'] > 0.5

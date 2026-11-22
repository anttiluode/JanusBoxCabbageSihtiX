"""Checks for the texture-gated mixture and the bend-then-EQ writer."""
import numpy as np
import pytest
from jbx import bands as B
from jbx.data import edge_world
from jbx.writers import raw_features, dark_filtered, WRITERS
from jbx.moe import TextureGate, descriptors, moe_columns, moe_predict, post_eq_columns

SEL = WRITERS['flashlight + odd translator']


@pytest.fixture(scope='module')
def ys():
    return [B.degrade(x)[0] for _, x in edge_world(11, 3)]


def test_one_expert_is_the_plain_writer(ys):
    F = raw_features(ys[0])
    A = moe_columns(F, SEL, np.ones((1, B.M, B.M)))
    assert np.allclose(A, dark_filtered(F, SEL), atol=1e-12 * np.abs(A).max())


def test_gates_partition_unity_and_ignore_contrast(ys):
    tg = TextureGate(4).fit(ys, per_image=1500)
    g1, g2 = tg.gates(ys[1]), tg.gates(5.0 * ys[1])
    assert np.allclose(g1.sum(0), 1)
    assert np.abs(g1 - g2).max() < 1e-3
    d1, d2 = descriptors(ys[2]), descriptors(0.2 * ys[2])
    assert np.abs(d1 - d2)[:, 24:232, 24:232].max() < 1e-3


def test_predict_matches_columns(ys):
    tg = TextureGate(3).fit(ys, per_image=1500)
    F, g = raw_features(ys[0]), tg.gates(ys[0])
    w = np.random.default_rng(1).normal(size=3 * len(SEL))
    p = B.interior(B.crop(moe_predict(F, SEL, g, w))).ravel()
    assert np.allclose(p, moe_columns(F, SEL, g) @ w, atol=1e-10 * np.abs(p).max())


def test_post_eq_even_tents_sum_to_dark_filter(ys):
    F = raw_features(ys[0])
    A = post_eq_columns(F, ['tri2'])
    assert np.allclose(A[:, :4].sum(1), dark_filtered(F, ['tri2']).ravel(), atol=1e-12 * np.abs(A).max())

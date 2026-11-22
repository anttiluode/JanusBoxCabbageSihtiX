# JanusBoxCabbageSihtiX

**Flashlight + translator: can a writer put back image detail where a blurred input has none?**

Open `index.html` in a browser (offline, no install). It regrows the erased fine band of an image live, with weights trained in this repo, and has the kill switches built in.

![dark band](results/figure.png)

## The question

Blur an image so nothing above 0.25 cycles/pixel survives (the **dark zone**). Every writer in the SighImageX family is a Fourier gain on the input: magnitude-only. A gain times zero is zero, so **no magnitude-only writer can put anything in the dark zone**. This is exact, not empirical (`tests/test_core.py::test_any_fourier_gain_writes_nothing_dark`).

To write there you must make new frequencies, which means a nonlinear writer. The proposal split it in two:

* **Flashlight** decides *where*: the unsigned amplitude of the octave below (the parent band, |p|) and the cross-scale phase agreement between parent and grandparent, `pc = |p+q| / (|p|+|q|)`.
* **Translator** decides *what*: a pointwise bend of the parent coefficient whose harmonics land in the dark zone.

Janus Cabbage had phase as the address and Sihti had the bands. The missing part was an outside reader that can tell a fixed filter from something that writes. This repo includes one.

## The sign trap (found while building)

The obvious translator is phase doubling, `p²/|p|`: it moves the parent's clock hand to the next octave. Together with `|b|`, it is **even**: negate the image and the feature stays the same. The detail it has to write is **odd**: negate the image and the fine band flips. Averaged over a world that contains both signs, an even writer's best weight is zero. The translator has to be odd, for example `clip(b)`, `sign(b)·b²`, or phase *tripling* `p³/|p|²`.

Every feature is homogeneous of degree 1, so contrast does not change what any writer does.

## Ledger

Dark-band R² on held-out images (`results/receipt.json`). 0 means "wrote nothing"; negative means it wrote the wrong thing. Training: 69 photo crops from 16 scikit-image photos; 60 synthetic edge scenes (anti-aliased shaded shapes). Test: 11 crops from 4 photos never used in training (astronaut, coffee, chelsea, grace_hopper); 20 edge scenes from a different seed; phase-scrambled twins of both.

| writer | trained on | edge test | edge twins | photo test | photo twins |
|---|---|---|---|---|---|
| any Fourier gain (SighImageX family) | – | **0** (exact) | 0 | 0 | 0 |
| even translator (\|b\|, p²/\|p\|, gated) | edge | −0.002 | −0.002 | −0.001 | −0.002 |
| odd translator, no flashlight | edge | +0.432 | −0.089 | −0.068 | −0.072 |
| tripled clock hands only (p³/\|p\|², gated) | edge | +0.661 | −0.412 | −0.194 | −0.332 |
| **flashlight + odd translator** | edge | **+0.746** | −0.586 | −0.205 | −0.475 |
| flashlight + odd translator | photo | +0.312 | −0.014 | **+0.053** | −0.011 |
| flashlight + odd translator | photo + edge | +0.680 | −0.181 | −0.005 | −0.142 |

Controls (flashlight + odd, edge-trained, on edge test): parent features rolled by 64×96 px → **−0.559**; parent features taken from the next image → **−0.762**. Photo-trained on photo test: −0.041 and −0.039.

The whole fine band (transition zone, where a radial EQ works, plus the dark zone): the dark zone holds 60% of the erased energy on edge scenes and 63% on photos.

| fine-band R² | edge test (edge weights) | photo test (photo weights) |
|---|---|---|
| blurred input | 0 | 0 |
| EQ on what survived (magnitude-only, 9-knot radial) | 0.380 | 0.354 |
| EQ + flashlight + odd translator | **0.829** | 0.387 |

The radial EQ alone gets R² ≈ 0.95 in the transition zone. That ceiling is set by the dark zone, not by the EQ.

### What this shows

1. **The sign trap is real.** The even translator scores 0.000 ± 0.002 everywhere, in every world. It is not "a weak translator"; it has the wrong symmetry.
2. **On edges, the split works.** The flashlight gate adds a lot on top of the odd translator (0.432 → 0.746). The largest weights in the edge-trained writer are `clip(b)·pc⁴` in all four orientations (weights 3.4–3.7, `results/weights.json`). That is Greenspan's clip, switched on only where the parent and grandparent phases agree.
3. **The kill test works.** Phase twins keep every Fourier magnitude and lose cross-scale phase alignment. The edge writer then does worse than writing nothing (−0.586). So it is using phase structure, not the spectrum. The same holds when the parent is displaced or swapped.
4. **On photographs it fails.** Trained on photos it reaches only +0.053. Trained on edges, or on edges plus photos, it is at or below zero on photos. With the mixed weights, 3 of the 11 photo crops are positive (best +0.30) and one astronaut crop reaches −0.85. Photos in this set are mostly texture, fur and soft shading, where the next octave is not a fixed function of the parent's local phase. A single linear readout over this dictionary cannot regrow them.
5. **Negation is a sanity check, not evidence.** Odd writers keep exactly the same score on the negative image by construction (`receipt.json → flip`). The even writer has nothing to lose.

## The outside reader

`jbx/reader.py` treats a writer as a sealed box. It sends inputs and reads outputs only. It runs four probes: dark-zone energy, additivity, negation parity, and residue-guided greedy dictionary recovery on held-out pixels (Luotain2 style).

| box | reader's verdict | additivity error | recovered |
|---|---|---|---|
| magnitude-only EQ | EQ / linear filter: nothing written where the input is dark | – | – |
| hidden tripler `0.6·tri0 − 0.4·triI2` | odd translator | 0.67 | exactly those two, weights 0.600, −0.400 |
| trained flashlight + odd (32 weights) | odd translator | 0.60 | all 32, rel. weight error 4e−13, residue 8e−15 |
| trained even writer | even translator | 0.71 | 24 terms, residue 6e−15 |
| \|p\| × fixed random phase | writes new frequencies, but outside the dictionary (climb higher) | 0.23 | unexplained fraction 1.0 |

Honest scope: recovering boxes that are built from the reader's own dictionary only checks that the reader is internally consistent. The useful verdict is the last row: it says when a box writes something this vocabulary cannot name, and it does not invent an explanation.

## Prior art (this is not new physics)

* Greenspan, Anderson & Akber, *Image enhancement by nonlinear extrapolation in frequency space*, IEEE TIP 2000: high-pass, clip, high-pass again. Our strongest single term is their clip. What is added here is the pc gate, the sign analysis and the controls.
* Portilla & Simoncelli 2000, texture model: phase-doubled parent coefficients correlated with child coefficients. The doubling is exactly where the sign problem lives. They use it as a statistic, not as a signed writer.
* Phase congruency (Morrone & Owens 1987; Kovesi 1999): features are where phases across scales agree. Our flashlight is a two-scale version of it.
* Complex steerable pyramids and phase-based motion magnification (Wadhwa et al. 2013): local phase as the clock hand.
* Learned super-resolution (SRCNN onward) and frequency-domain token mixers (AFFNet ICCV 2023, DFFormer AAAI 2024) do far better on photos. This repo does not compete with them. It is a small, readable, closed-form operator with the kill tests attached.

## Run it

Windows, Python 3.13:

```
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pytest -q tests            # 14 tests, ~20 s
py -3.13 experiment.py                 # ~4 min on CPU -> results/receipt.json, results/weights.json
py -3.13 moe_experiment.py             # ~50 min on CPU -> results/moe_receipt.json
py -3.13 figure.py                     # results/figure.png
py -3.13 build.py                      # index.html (embeds web/jbx.js, weights, 4 held-out crops)
node tests/test_js.js                  # browser core vs Python core (agree to ~1e-11)
```

On Linux or macOS use `python3` in place of `py -3.13`. No GPU is used. Everything is NumPy FFTs and normal equations.

## Round 2: let the flashlight choose the translator (`moe_experiment.py`)

The flashlight now also picks *which* translator to use. It reads five contrast-free texture measures from the blurred input, each a Gaussian local average with σ = 8 px:

* **agree:** phase agreement between parent and grandparent.
* **orient:** how oriented the patch is.
* **slope:** parent/grandparent energy slope.
* **level:** local energy relative to the whole image.
* **peaky:** edge versus texture, a 4th/2nd moment of |p|.

k-means on training pixels gives G texture centres, and each pixel gets soft gates. A second new writer, **bend then EQ**, gives every translator feature its own small dark-zone filter: 4 radial tents, each with a zero-phase version and a quadrature version along the feature's orientation. Both writers are still closed-form least squares. Receipt: `results/moe_receipt.json`. Run time: 48 min on 2 CPU cores.

Dark-band R² on the same held-out sets as above. Trained on photos unless marked.

| writer | weights | photo test | photo crops > 0 | edge test | photo twins | photo, parent shifted | photo, parent from next image |
|---|---|---|---|---|---|---|---|
| flashlight + odd (round 1) | 32 | +0.053 | 9/11 | +0.312 | −0.011 | −0.041 | −0.039 |
| texture-gated, G = 3 | 96 | +0.055 | 8/11 | +0.384 | −0.009 | −0.053 | −0.054 |
| texture-gated, G = 6 | 192 | +0.055 | 9/11 | +0.426 | −0.010 | −0.065 | −0.070 |
| texture-gated, G = 12 | 384 | +0.050 | 8/11 | +0.479 | −0.019 | −0.079 | −0.084 |
| G = 6, gates read from **another image** (capacity control) | 192 | +0.039 | 7/11 | +0.254 | −0.014 | −0.056 | −0.054 |
| **bend then EQ**, G = 1 | 256 | **+0.066** | 8/11 | +0.480 | −0.015 | −0.065 | −0.062 |
| bend then EQ, trained on photos + edges | 256 | +0.021 | 4/11 | **+0.767** | −0.088 | −0.243 | −0.251 |
| texture-gated G = 6, trained on photos + edges | 192 | −0.033 | 3/11 | +0.706 | −0.063 | −0.267 | −0.306 |

How much of a photo's dark band could a readout like this explain at all? To find out, fit on 2/3 of a test crop's own pixels and score the other 1/3. Medians over the 11 photo crops:

| within-image ceiling | median | range |
|---|---|---|
| flashlight + odd (32 weights) | 0.088 | 0.005 to 0.40 |
| texture-gated G = 12 (384) | 0.151 | 0.03 to 0.45 |
| bend then EQ (256) | 0.138 | 0.01 to 0.45 |
| every feature at 9 offsets ±3 px (288) | 0.171 | 0.02 to 0.47 |

### Verdict

1. **Texture gating does not move photos.** The score goes from 0.053 to 0.055 with 3 or 6 experts, and to 0.050 with 12. On edge scenes the same gating helps a lot (0.312 → 0.479), so the gates work, just not where the problem is. Texture gates beat gates read from another image (0.055 vs 0.039 on photos, 0.426 vs 0.254 on edges), so the texture measures carry real information. It is not just extra capacity, but on photos there is little to gain.
2. **Bend then EQ is the best photo writer**, at +0.066. It is also the best writer for mixed training: +0.767 on edges, and above zero on photos (+0.021) where round 1's mixed writer was at −0.005.
3. **The kill tests still hold.** Every writer falls below zero on phase twins, on a shifted parent and on a parent taken from another image. The gate-shift control (`receipt.json`) collapses to between −17 and −520. The experts use large opposite-sign weights that only cancel when the gate sits on the right texture, so the gated writers are fragile.
4. **So, is a photo's dark zone recoverable from one octave below?** Not by local readouts of the parent and grandparent. Fitted on each photo's own pixels with 288 weights, they still explain a median of only 17% of the dark band. Out of sample the best is 7%. Most of the detail above 0.25 cycles/pixel in these photos (fur, hair, grain, foliage, sensor noise) is not a fixed function of the local analytic coefficients one octave down. What this does *not* show: that the detail is unrecoverable by any method. Learned super-resolution uses wide context and priors learned from millions of images, which is a different instrument. This ledger only covers the flashlight and translator family, and for photos that family has reached its ceiling.

## Files

* `jbx/bands.py`: the fixed geometry. 256→512 mirror extension, fine, parent and grandparent bands, dark zone, K = 4 one-sided steerable windows.
* `jbx/writers.py`: the feature dictionary (odd, even, gated), the radial EQ basis, Gram accumulation.
* `jbx/moe.py`: texture descriptors, soft k-means gates, gated columns, and bend-then-EQ filters.
* `jbx/reader.py`: the outside reader.
* `jbx/data.py`: photo crops, edge world, phase twins.
* `experiment.py`: round 1. Train, test, controls, reader → `results/receipt.json`, `results/weights.json`.
* `moe_experiment.py`: round 2 → `results/moe_receipt.json`, `results/moe_weights.json`.
* `web/jbx.js` + `web/template.html` → `index.html`: the live page (round 1 writers). Buttons: negate, phase twin, shift parent, choice of training world, load your own image.

Built by Claude (Opus 5.5) on the bands from SighImageX, BlackBoxSpotlight (Astra) and SighImageInverse (Sol).

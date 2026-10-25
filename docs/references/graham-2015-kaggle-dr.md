# Notes: Ben Graham, Kaggle Diabetic Retinopathy Detection report (2015)

Source: B. Graham, "Kaggle Diabetic Retinopathy Detection competition report", University of
Warwick, Aug 2015. First place (team Min-Pooling), private-leaderboard quadratic weighted kappa
0.84958 on the EyePACS test set. Code: github.com/btgraham/SparseConvNet (GPL v3, branch
`kaggle_Diabetic_Retinopathy_competition`); we reimplement ideas, we do not copy code.

## What he did

| Step | His choice | Notes |
|---|---|---|
| Rescale | Scale each image so the fundus **radius** is 300 or 500 px (radius estimated from the middle row: pixels brighter than mean/10) | Radius-based, not bounding-box based |
| Normalise | `4·img − 4·GaussianBlur(img, σ = radius/30) + 128` (local mean → 50 % grey) | The "Ben Graham" preprocessing in PLAN.md §8.1 |
| Border | Circular mask at **90 % of the radius**, outside set to 128 | Removes the bright ring the blur creates at the fundus edge |
| Augment (train) | Scale ±10 %, rotate 0–360°, skew ±0.2 | Colour-space augmentation **did not help** (he suspects because preprocessing already removes local colour) |
| Augment (test) | Random rotations, averaged | Test-time augmentation (TTA) |
| Models | 3 CNNs (two fractional max-pooling, one VGG-style), 5-class softmax | Ensemble only marginally better than the best single model |
| Resolution | Radius 270 used; **going above 270 slowed things down with no benefit** | |
| Two-eye fusion | Random forest on: own-eye probabilities, **other-eye probabilities**, raw image size, raw and preprocessed variance | Linear regression on the two eyes' probabilities alone "also seems to work well" |
| Kappa thresholds | Continuous score cut at {0.57, 1.37, 2.30, 3.12} | Specific to quadratic kappa |
| Validation | 1,000 images | He calls it "quite small" |

## What to adopt in RetinaGuard

1. **Blur σ: use `radius/30`, not `512/30`.** His σ is tied to the fundus radius. At 512 px
   the radius is ≈ 256 px, so σ ≈ 8.5. PLAN.md §8.1 currently says `σ = 512/30` ≈ 17, which
   blurs twice as much as the original. Fix when implementing `preprocess.py`.
2. **Mask at 0.9 × radius after normalisation**, fill 128. Without it the blur leaves a
   bright edge ring that a CNN can latch onto.
3. **Keep 512 px.** His "no benefit above radius 270" result supports the 512 px default
   (radius ≈ 256) in CLAUDE.md. Use it as evidence in the image-size ADR rather than going larger.
4. **Rotation 0–360° plus ±10 % scale** (already in §8.2), and **add small shear (±0.2)**.
5. **Colour augmentation becomes part of the Ben Graham ablation**: expected to help
   *without* the normalisation and little *with* it. Test both rather than assume.
6. **TTA with rotations** at evaluation time is cheap. Report results with and without TTA, and
   measure the extra inference latency for the API.
7. **Skip large ensembles.** His ensemble gave marginal gains. Our Kaggle GPU budget is
   better spent on the backbone comparison (Phase 3) and ablations.
8. **Two-eye fusion is a strong, cheap signal**: both eyes of a patient are highly correlated.
   Candidate extension, with three conditions:
   - fit the fusion model on **val** predictions, never on train predictions. He notes his random
     forest was trained on outputs for images the CNN had already seen, a train/test mismatch
     that "luckily" did not overfit;
   - recalibrate and recompute conformal thresholds **after** fusion, since fusion changes the
     score distribution;
   - the API must accept a single image and fall back to per-eye scoring, because not every
     request will include both eyes.
9. **Do not feed raw image size or variance into the model.** Those are camera and site cues.
   They helped on a single-site leaderboard but are shortcut features that hurt external sites,
   which is exactly what RetinaGuard evaluates.
10. **Use a larger validation set than his.** Ours (10 % of patients, about 3.5k images) plus bootstrap CIs
    avoids the small-val noise he mentions.

## Comparing our results with his

- His 0.85 is **5-class quadratic weighted kappa on the official EyePACS test set**. Our
  `test_internal` is a 10 % patient split of the labelled train set, so numbers are comparable in
  spirit only. Report QWK on `test_internal` alongside our main metrics (referable AUROC,
  sensitivity at fixed specificity, FNR under deferral) and state the difference.
- If kappa thresholds are optimised, fit them on `val`, never on a test set.
- "Similar or better" for RetinaGuard means more than kappa: it also means calibrated
  probabilities, a bounded FNR with deferral, and honest per-site external results.

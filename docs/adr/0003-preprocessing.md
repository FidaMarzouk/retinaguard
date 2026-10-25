# 0003: Fundus preprocessing and Ben Graham normalisation

- Status: accepted
- Date: 2026-10-25

## Context
PLAN.md §8.1 asks for circle crop, 512 px resize and an optional Ben Graham normalisation with a
with/without ablation. The original method (Graham, Kaggle DR 2015; notes in
`docs/references/graham-2015-kaggle-dr.md`) ties the blur width to the fundus radius and masks
the outer 10 % of the disc.

## Decision
- Stored processed images are **crop + resize only** (`crop512_v1`). Ben Graham normalisation is
  applied on the fly at load time, so one processed dataset serves both ablation arms.
- Blur sigma = fundus radius / 30 (≈ 8.5 px at 512 px), not 512 / 30 as first written in §8.1.
- After normalisation, pixels outside 0.9 × radius are set to grey (128).
- Fundus detection: largest region whose red channel exceeds max(0.1 × mean red, 10); radius from
  the region's widest extent so discs clipped by the frame keep their true radius. Images without
  a plausible disc raise `FundusNotFoundError`, which preprocessing records instead of dropping
  silently.
- 512 px stays the default; Graham saw no gain above a 270 px radius.

## Consequences
Training pays one Gaussian blur per image when the ablation arm uses Ben Graham, which is cheap at
512 px. Changing crop or resize logic requires a new `PREPROCESS_VERSION`.

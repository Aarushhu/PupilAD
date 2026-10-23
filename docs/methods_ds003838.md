# ds003838 methods

All parameters were set before the group results were looked at, and match the phase-1 run of
12 Oct 2026. They live as named constants in `src/pupilad/ds003838/preprocess.py` and
`src/pupilad/ds003838/features.py`.

## Data

- OpenNeuro ds003838 v1.0.6. 86 participants, 84 with pupil data (sub-017 and sub-094 have none).
- Pupil Labs Core, 120 Hz, `diameter_3d` (mm) from the `3d c++` model, both eyes.
- Task: auditory digit span, digits every 2 s. Memory condition (memorize, then recall) and
  control condition (just listen) at 5, 9 and 13 digits. 162 sequences per person
  (108 memory, 54 control) in 9 blocks of 18.
- NASA-TLX was given three times, after blocks 3, 6 and 9. Each administration covers a mix of all
  three loads, so TLX cannot be matched to a specific load.

## Preprocessing (per eye)

1. Mark samples invalid if confidence < 0.6, blink flag = 1, or diameter outside 1.5–9 mm.
2. Extend every invalid stretch by 100 ms on each side.
3. Resample valid samples onto a uniform 120 Hz grid with linear interpolation; gaps longer than
   500 ms stay missing.
4. Zero-phase 4th-order Butterworth low-pass at 4 Hz, then keep every 6th sample (20 Hz).
5. Use the dominant eye unless the other eye has at least 15 percentage points more valid data.

## Epochs and features

- One epoch per digit sequence, from 0.5 s before the first digit to 2 s after the last.
- Baseline = mean of the 0.5 s before the first digit; all values are change from baseline (mm).
- Per-digit value = mean in each 2 s window after each digit onset.
- End-of-sequence value = the window of the last digit.
- A trial is rejected if more than 30% of its samples were originally invalid or the sequence is
  incomplete. A person is excluded if fewer than 54 memory or 27 control trials survive.

Per-person features (`subject_features.csv`):

| Feature | Definition |
|---|---|
| `mem_minus_ctrl_end_{5,9,13}` | memory minus control end-of-sequence dilation |
| `mem_minus_ctrl_sequence_mean_{L}` | memory minus control, averaged over all digit windows |
| `load_slope_end_mm_per_digit` | least-squares slope of `mem_minus_ctrl_end` over load 5, 9, 13 |
| `load_quadratic_contrast_end` | 5-digit − 2 × 9-digit + 13-digit effect; negative means inverted U |
| `sequence_slope_mem_minus_ctrl_{L}` | within-sequence slope over digits 1…L, memory minus control |
| `encoding_slope_memory` | slope over digits 1–5, all memory trials pooled |
| `slope_digits_1to7_load13`, `slope_digits_8to13_load13` | 13-digit memory slopes before and after typical capacity |
| `tlx_{mental,effort,raw}_{1,2,3,mean}` | NASA-TLX mental demand, effort, and mean of all six scales |

## Statistics

- Paired effects: mean with 95% bootstrap CI (10,000 resamples, seed 2027), Cohen's dz with
  bootstrap CI, rank-biserial correlation and Wilcoxon signed-rank p.
- Correlations: Spearman ρ with bootstrap CI.
- Within-person TLX test: repeated-measures correlation across the three TLX administrations
  (CI from a Fisher-z approximation).
- Holm correction within each family of tests.

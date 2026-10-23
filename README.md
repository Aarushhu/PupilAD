# PupilAD

PupilAD is a pupillometry pipeline built for an ISEF 2027 project. It measures how much the pupil
dilates when a person holds information in working memory (task-evoked pupillary response) and
turns that into per-person features.

The long-term goal is to study whether these features could become a low-cost marker of
locus coeruleus (LC) function, which is affected early in Alzheimer's disease. **The current
evidence does not support any diagnostic claim.** See [Claim boundaries](#claim-boundaries).

## What has been tested

All real-data results come from one public dataset, OpenNeuro
[ds003838](https://openneuro.org/datasets/ds003838) (Pavlov et al., 2022): healthy adults aged
18–25 doing an auditory digit-span task with 5, 9 or 13 digits, recorded with a Pupil Labs Core
head-mounted infrared tracker at 120 Hz.

- Main results: [`results/literature_comparison.md`](results/literature_comparison.md)
- Every statistic: [`results/ds003838/stats.json`](results/ds003838/stats.json)
- Figures: [`results/ds003838/`](results/ds003838/)
- Synthetic-data checks of the LSTM: [`results/synthetic/`](results/synthetic/)

## Claim boundaries

What the ds003838 analysis **can** support:

- **Measurement validity**: the preprocessing produces plausible pupil traces and reliable
  per-person features (split-half reliability is reported).
- **Construct validity for cognitive load**: memorizing produces more dilation than just listening,
  and the effect changes with the number of digits in the way published work predicts.
- **Consistency with the literature**: direction and rough size of effects compared with published
  studies, with every disagreement reported.

What it **cannot** support:

- Anything about Alzheimer's disease, MCI, tau, amyloid or LC integrity. Every participant was a
  healthy young adult, and no LC measure (for example neuromelanin MRI) exists in this dataset.
- Anything about webcam pupillometry. ds003838 used an infrared, head-mounted tracker.
- The 0–100 "LC integrity score" and its 70/50 risk tiers. The LSTM has only been trained on
  synthetic data whose two classes were designed by hand, so its accuracy there shows the code
  runs, not that the score means anything clinically.
- Published AUCs from other studies (for example 0.749 for neuromelanin MRI of the LC) are
  reference points for future work, not results of this project.

## Reproduce

Requires Python 3.11+, about 25 GB of free disk space during download (each raw pupil file is
about 2 GB and is deleted after filtering) and a few hours of download time.

```bash
python -m venv .venv
make install            # pinned requirements + editable install of the pupilad package
make download           # ds003838 pupil, events and behaviour files into data/ (gitignored)
make preprocess         # clean, resample and epoch each participant
make features           # per-person features and inclusion table
make validate           # statistics and figures into results/ds003838/
make train              # synthetic-data recovery check for the LSTM (needs TensorFlow)
make test               # unit tests
```

Each step is also available as `python -m pupilad <step>`. Other commands: `lc-check`,
`tobii-features`, `plr-features`, `extract-webcam`, `webcam-trials`.

The PsychoPy task battery needs its own environment because PsychoPy pins different versions:

```bash
python -m venv .venv-psychopy
.venv-psychopy/bin/pip install -r requirements-battery.txt
.venv-psychopy/bin/pip install -e . --no-deps
.venv-psychopy/bin/python scripts/run_task_battery.py
```

## Repository layout

```
src/pupilad/
  ds003838/      download, preprocess, features, validate, figures for the public dataset
  tobii/         Tobii Pro Lab export ingest and synthetic export generator
  plr/           pupillary light reflex ingest
  webcam/        webcam pupil detector, video extraction, trial table builder
  model/         LSTM definition and risk-tier mapping
  validation/    LC-score sanity checks, literature benchmarks, synthetic recovery experiment
  signal.py      shared signal cleaning (artifact masks, gap interpolation, low-pass)
  stats.py       effect sizes, bootstrap intervals, Holm correction, repeated-measures correlation
scripts/         task battery, webcam session, download helper
tests/           unit tests for preprocessing, load slope and statistics
docs/            methods notes and file provenance
results/         committed outputs
data/            downloaded and derived data (gitignored)
notebooks/       ADNI exploration (needs ADNI data access)
```

Processing parameters for ds003838 are listed in [`docs/methods_ds003838.md`](docs/methods_ds003838.md).

## Original work

Designed for this project:

- The study question and the overall pipeline design: task battery → pupil features → per-person
  score, with the validation plan split into measurement validity, construct validity and
  literature comparison.
- The PsychoPy task battery (`scripts/run_task_battery.py`) and webcam session
  (`scripts/webcam_task_session.py`).
- The ds003838 analysis plan: inclusion rules, load-slope features, NASA-TLX comparison.
- The synthetic data generator and the synthetic recovery experiment for the LSTM.
- The ADNI CSF reference summary in `src/pupilad/validation/benchmarks.py`, computed from a local
  ADNI extract.

Much of the code was written with help from AI assistants (Claude). The provenance of each
file, including which ones look adapted from other sources, is recorded in
[`docs/provenance.md`](docs/provenance.md).

## Third-party components

**Datasets**

| Dataset | Use | License / terms |
|---|---|---|
| OpenNeuro ds003838 v1.0.6 — Pavlov, Kasanov, Kosachenko, Kotyusov & Busch (2022), *Scientific Data* 9:325, doi:10.1038/s41597-022-01414-2 | All real-data results | CC0 |
| ADNI (UPENNBIOMK_MASTER, DXSUM, PTDEMOG, MODHACH) | CSF reference values, exploration notebook | ADNI Data Use Agreement; not redistributable |

**Libraries**: NumPy, pandas, PyArrow, SciPy, statsmodels, Matplotlib, scikit-learn and pytest
(BSD-3-Clause, MIT or Apache-2.0); TensorFlow/Keras (Apache-2.0); OpenCV (Apache-2.0); PsychoPy
(GPL-3.0, used only by the separate task-battery scripts); black and isort (MIT). Exact versions
are pinned in `requirements.txt` and `requirements-battery.txt`.

**Models and data files from libraries**: OpenCV's bundled Haar cascades
`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` (Intel License Agreement, shipped
with OpenCV) are used by the webcam pupil detector.

**Adapted code**: see [`docs/provenance.md`](docs/provenance.md).

## License

Code in this repository is released under the MIT License (see `LICENSE`). Datasets keep their own
licenses and are not included in the repository.

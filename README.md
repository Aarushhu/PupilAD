# PupilAD

PupilAD measures how much the pupil widens when a person holds information in working memory.
This is the task-evoked pupillary response. The code turns that response into per-person features.
The long-term aim is a low-cost marker of locus coeruleus function, which changes early in
Alzheimer's disease. **The current results do not support any diagnostic claim**

The repository has three parts:

1. **Analysis of a public dataset.** It cleans pupil recordings from OpenNeuro
   [ds003838](https://openneuro.org/datasets/ds003838) and tests whether pupil size tracks memory
   load.
2. **A task battery app.** It runs four memory and attention tasks on a laptop, phone or iPad while
   the camera records the participant's eyes, then extracts pupil size from the video.
3. **Support tools.** These read Tobii and pupillary light reflex exports, and include an LSTM model
   so far tested only on synthetic data.

## Task battery app

| Task                                            | Measures                           |
| ----------------------------------------------- | ---------------------------------- |
| Letter Match (2-back)                           | working memory updating            |
| Number Recall (digit span, forward and reverse) | working memory capacity            |
| Color Clash (Stroop)                            | attention and interference control |
| Animal Sprint (verbal fluency)                  | word retrieval, recorded as audio  |

The app checks camera framing and lighting before each session. Participants answer by tapping or
by using the keyboard, and each task has an optional practice round. The video is recorded in the
browser. After the session the app automatically saves the scores, a pupil-size trace and a
per-trial pupil table. Details, output files and timing accuracy are in [`docs/app.md`](docs/app.md).

## Where everything is

### Suggested reading order

1. [`results/literature_comparison.md`](results/literature_comparison.md): the main results, with a
   short "Bottom line" at the start and a section on what they cannot show.
2. [`results/ds003838/`](results/ds003838/): the figures, and `stats.json` with every statistic.
3. [`docs/methods_ds003838.md`](docs/methods_ds003838.md): data, preprocessing, features and
   statistics, with all parameters.
4. [`src/pupilad/ds003838/`](src/pupilad/ds003838/): the code behind those results. Start with
   `preprocess.py`, then `features.py`, then `validate.py`.
5. [`docs/app.md`](docs/app.md), then [`src/pupilad/app/`](src/pupilad/app/): how the task battery
   app works and the code behind it.

### Finding a specific thing

| Looking for | Go to |
|---|---|
| Main findings | `results/literature_comparison.md` |
| A specific statistic | `results/ds003838/stats.json` |
| Time courses of dilation by load | `results/ds003838/fig1_load_dilation.png` |
| Split-half reliability and overload | `results/ds003838/fig2_reliability_overload.png` |
| Load slope and NASA-TLX | `results/ds003838/fig3_load_slope_tlx.png` |
| Preprocessing and feature parameters | Named constants in `src/pupilad/ds003838/preprocess.py` and `features.py`; explained in `docs/methods_ds003838.md` |
| Participant inclusion rules | `mark_inclusion` in `src/pupilad/ds003838/features.py` |
| Statistical tests | `src/pupilad/ds003838/validate.py`, using helpers from `src/pupilad/stats.py` |
| Published values used for comparison | `src/pupilad/validation/benchmarks.py` |
| Tasks shown to participants | `src/pupilad/app/static/battery.js` (app) and `scripts/run_task_battery.py` (older PsychoPy version) |
| Task scoring | `src/pupilad/app/scoring.py` |
| How pupil size is found in video | `src/pupilad/webcam/pupil_detector.py` |
| App output files, timing and consent notes | `docs/app.md` |
| Risk tiers and LSTM definition | `src/pupilad/model/lstm.py` (not clinically validated) |
| All command names | `COMMANDS` in `src/pupilad/__main__.py` |
| All data and result paths | `src/pupilad/paths.py` |

### How the ds003838 analysis fits together

Each step reads the output of the step before it. Everything under `data/` is created on your
machine and is not on GitHub.

| Command | Code in `src/pupilad/ds003838/` | Writes to |
|---|---|---|
| `make download` | `download.py` | `data/ds003838/raw/`, `data/ds003838/slim/` |
| `make preprocess` | `preprocess.py` (with `events.py`, `loaders.py`, `../signal.py`) | `data/ds003838/derived/trials/` |
| `make features` | `features.py` | `data/ds003838/derived/*.csv` |
| `make validate` | `validate.py`, `figures.py` (with `../stats.py`) | `results/ds003838/` |

### File map

```
README.md                  this overview and map
Makefile                   shortcuts for each analysis step
pyproject.toml             package definition
requirements.txt           pinned versions for the analysis
requirements-app.txt       packages for the task battery app
requirements-battery.txt   pinned versions for the PsychoPy task battery
start_app.bat              start the app on a Windows computer
start_app_phone.bat        start the app for phones and tablets on the same Wi-Fi

docs/
  methods_ds003838.md      data, preprocessing, features and statistics
  app.md                   using the app, output files, timing, privacy and consent

results/
  literature_comparison.md write-up of results against published studies
  ds003838/                stats.json and the three figures
  synthetic/               saved LSTM recovery runs on synthetic data only

src/pupilad/
  __main__.py              command-line entry point (python -m pupilad <step>)
  paths.py                 data and results locations
  signal.py                artifact masks, gap interpolation, low-pass filter
  stats.py                 effect sizes, bootstrap intervals, Holm correction,
                           repeated-measures correlation
  ds003838/                public dataset analysis (all real-data results come from here)
    download.py            fetch pupil, events and behaviour files
    events.py              decode event labels into trials, conditions and digits
    loaders.py             read raw and derived files
    preprocess.py          clean, resample and epoch each participant
    features.py            inclusion rules and per-person features
    validate.py            statistics, written to stats.json
    figures.py             the three result figures
  app/                     task battery app
    server.py              web server, sessions, saved files, automatic analysis
    scoring.py             task score summaries
    checks.py              camera check on a single frame
    recording.py           convert the browser recording to video and frame timestamps
    tls.py                 HTTPS certificate for phones and tablets
    static/                browser side
      index.html, app.css  page and styling
      app.js               screens and session flow
      battery.js           the four tasks and their settings
      media.js             camera, microphone, recording and upload
      chart.js             pupil trace chart on the Results tab
  webcam/
    pupil_detector.py      OpenCV pupil detection for one video frame
    video_extraction.py    pupil trace from a recorded session video
    trial_table.py         combine a pupil trace with task logs
    column_map.json        column settings for webcam data
  tobii/
    ingest.py              features from Tobii Pro Lab exports
    synthetic.py           synthetic Tobii-style export generator
  plr/
    ingest.py              pupillary light reflex features
  model/
    lstm.py                LSTM definition and score-to-risk-tier mapping
  validation/
    benchmarks.py          published reference values and ADNI CSF summary
    lc_score.py            sanity checks on task pupil features (lc-check)
    synthetic_recovery.py  LSTM recovery experiment on synthetic data (train)

scripts/
  run_task_battery.py      older PsychoPy task battery (separate environment)
  webcam_task_session.py   webcam task session with live pupil recording
  download_ds003838.sh     wrapper around the download step

tests/
  test_signal.py           signal cleaning helpers
  test_ds003838_preprocess.py  event parsing, eye cleaning, trial epoching
  test_load_slope.py       load-slope features, statistics, risk tiers
  test_app.py              app sessions, scoring, uploads, access key

notebooks/
  01_adni_exploration.ipynb  ADNI exploration (needs ADNI data access)

data/                      local only, gitignored
  ds003838/                raw/, slim/ and derived/ analysis files
  adni/                    ADNI tables, not redistributable
  task_battery_logs/       session files from the app and the PsychoPy battery
  webcam_pilot/            local webcam pilot data
  app_certificate/         certificate created for phone and tablet mode
```

All real-data statistics come from ds003838. The Tobii, PLR, webcam and LSTM code has no real-data
results in this repository, and the committed LSTM results in `results/synthetic/` come from
hand-designed synthetic data. Running `make train` writes a new file,
`results/synthetic/recovery_scaling.csv`, and leaves the saved runs unchanged.

# File provenance

This list records where code and numbers came from, so the author can decide how to credit them.
Nothing listed here was rewritten to hide its origin. Items marked **Decide** need the author's
call.

## Code that looks adapted from elsewhere

| File | Origin | Notes |
|---|---|---|
| `src/pupilad/ds003838/preprocess.py`, `features.py`, `validate.py`, `figures.py` | Restructured from the phase-1 pipeline (`process_subject.py`, `analysis.py`, `batch.py`) generated in an earlier AI-assistant chat (Oct 2026, stored outside the repo in `PupilAD Chat Attachments/ds003838_phase1_results/pipeline`). | Same parameters and outputs (sub-013 QC reproduces exactly). **Decide**: how to disclose AI-generated analysis code. |
| `src/pupilad/webcam/pupil_detector.py` (was `src/pupil_tracker.py` and duplicated in `src/task_display.py`) | Haar-cascade face → eye → dark-threshold → largest contour pattern. This is a very common OpenCV tutorial pattern; no specific source was recorded. | Uses OpenCV's bundled cascade XML files (Intel License Agreement). **Decide**: whether it was copied from a tutorial and should cite it. |
| `scripts/run_task_battery.py` | PsychoPy API usage follows PsychoPy documentation. The original file header said it was written but "NOT runtime-tested" by its author, which reads like AI-assistant output. | **Decide**: AI-assistance disclosure. |
| `src/pupilad/tobii/ingest.py`, `src/pupilad/plr/ingest.py`, `src/pupilad/model/lstm.py`, `src/pupilad/tobii/synthetic.py` | Original headers were labelled "PupilAD 2.0 — Stage 2…" and written in an assistant voice. `PupilAD.py` at the repo root was an older copy of the Tobii ingest (removed as a duplicate). | **Decide**: AI-assistance disclosure. |
| `src/pupilad/validation/lc_score.py`, `src/pupilad/validation/synthetic_recovery.py` | The original header of `validate_lc_score.py` contained first-person assistant text ("Where I only have a directional finding…"). | **Decide**: AI-assistance disclosure. |
| `src/pupilad/app/` (web app), `start_app.bat`, `tests/test_app.py`, `docs/app.md` | Written with an AI coding assistant (Nov 2026). The task logic and instruction text are ported from `scripts/run_task_battery.py`; output files use the same names and columns so `extract-webcam` and `webcam-trials` work unchanged. Added an optional `progress` argument to `webcam/video_extraction.extract_pupil_trace`. | **Decide**: AI-assistance disclosure. |
| Refactor of all files in this repository (Oct 2026) | Done with an AI coding assistant: comments and docstrings removed, names clarified, modules split. Outputs checked against the previous versions on identical inputs. | Recorded in commit history with a co-author trailer. |

## Numbers that need a source

| Location | Value | Status |
|---|---|---|
| `validation/benchmarks.py` `constriction_velocity_cutoff_high/low` | 4.43 and 3.12 mm/s | **UNVERIFIED** — original citation was only "Constriction-velocity diagnostic cutoff study". |
| `validation/benchmarks.py` `pupil_size_change_cutoff` | 1.77 mm, sens 0.613, spec 0.80, PPV 0.4776, NPV 0.8688 | **UNVERIFIED** — original citation was only "Pupil-size-change cutoff study". |
| `validation/benchmarks.py` `dilation_amplitude_tau_association` | Granholm et al. 2021, tau not amyloid | **UNVERIFIED** — not checked in this pass. |
| `model/lstm.py` risk tiers | 70 / 50 cut-offs | No source; design choice with no validation. |
| Jang et al. 2021, AUC 0.71 | Reference point named in the project plan | **UNVERIFIED** — no matching paper found. |

## Data in git history

Four ADNI tables (`PupilAD-ISEF2027/data/adni/*.csv`) were committed in July–September 2026.
On 26 Oct 2026 they were purged from every commit with `git filter-repo` and the cleaned history
was force-pushed, because ADNI's Data Use Agreement does not allow redistribution. Commit dates and
messages were kept; commit IDs changed. ADNI data now lives only on the author's machine and is
excluded by `.gitignore` (`data/`).

## Removed as duplicates or dead code

- `PupilAD-ISEF2027/` — nested older copy of the whole project.
- `PupilAD.py` — older copy of the Tobii ingest script.
- `src/task_display.py.save` — editor backup.
- Two commented-out earlier versions of the task battery at the bottom of `run_task_battery.py`.
- `train_on_synthetic()` in the LSTM script (demo on a CSV that was never generated; superseded by
  `pupilad train`) and the live-session loop in `src/pupil_tracker.py` (superseded by
  `scripts/webcam_task_session.py`).

All of these remain available in git history.

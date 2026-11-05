# File provenance

This list records where code and numbers came from, so the author can decide how to credit them.
Nothing listed here was rewritten to hide its origin. Items marked **Decide** need the author's
call.

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

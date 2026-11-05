# PupilAD app

A local web app that runs the cognitive task battery with webcam recording, then scores the
tasks and extracts the pupil trace. It replaces `scripts/run_task_battery.py` for day-to-day use;
that script still works and writes the same files.

## Start

- Windows: double-click `start_app.bat`. The first run creates `.venv-app` and installs
  `requirements-app.txt`.
- Any OS: `pip install -r requirements-app.txt`, then `PYTHONPATH=src python -m pupilad app`.

The app opens at <http://127.0.0.1:5050/>. Options: `--port`, `--camera N`, `--logs_dir PATH`,
`--no-browser`. It only listens on this computer and has no login, so don't expose the port to
a network.

## Running a session

1. Enter the subject ID (letters, digits, `-`, `_`). If that ID already has data, you must tick
   "Overwrite" or pick a new ID.
2. Check the camera preview. "Pupil detected" should be above about 50%.
3. Choose the tasks. Defaults match `run_task_battery.py`; change them under "Advanced settings".
   Settings are remembered in the browser and saved with each session.
4. Press **Start session**. The battery goes full-screen. **Shift+Esc** aborts; tasks already
   finished are kept.

## Output

Files go to `data/task_battery_logs/` with the same names and columns as the PsychoPy script:
`{id}_markers.csv`, `{id}_nback.csv`, `{id}_digitspan.csv`, `{id}_stroop.csv`, `{id}_webcam.mp4`,
`{id}_webcam_frame_timestamps.csv`. The app also writes `{id}_session.json` with the settings,
start and end times, webcam frame rate, browser, and clock-sync quality.

On the Results tab, **Extract pupil data from video** runs the same code as
`pupilad extract-webcam` and `pupilad webcam-trials`, producing `{id}_pupil_trace.csv` and
`{id}_trial_pupil_table.csv`.

## Timing

Stimuli are drawn in the browser and the webcam is recorded by Python. Both are put on one clock:
at the start and end of each session, the page measures its offset from the server clock using
the fastest of 15 round trips. Each onset is stamped at the display frame in which the stimulus is
drawn, so it carries about one refresh (~17 ms at 60 Hz) of uncertainty. That is well under one
webcam frame (33 ms). `session.json` records the sync error and the drift over the session.

## Differences from `run_task_battery.py`

- N-back letters stay on screen for the full 0.5 s and the next letter comes exactly 2 s after
  the previous one. In the PsychoPy version a key press removed the letter early and pushed later
  letters back. As before, only presses while the letter is visible are recorded.
- Tasks can be switched off and parameters changed per session.
- Each task's CSV is saved as soon as that task ends, so an aborted session keeps finished tasks.
- On Windows the camera opens with DirectShow, which starts in about 1 s instead of about 15 s.

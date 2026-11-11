# PupilAD app

A web app that runs the cognitive task battery in the browser on a laptop, phone or iPad. It
records the front camera during the tasks, then scores them and extracts the pupil trace. It
replaces `scripts/run_task_battery.py` for day-to-day use; that script still works and writes the
same files.

## Start

- Windows, on this computer: double-click `start_app.bat`. The first run creates `.venv-app` and
  installs `requirements-app.txt`.
- Windows, for phones and tablets: double-click `start_app_phone.bat` (see below).
- Any OS: `pip install -r requirements-app.txt`, then `PYTHONPATH=src python -m pupilad app`.

The app opens at <http://127.0.0.1:5050/>. Options: `--port`, `--logs_dir PATH`, `--no-browser`,
`--lan`.

## Portable pack (flash drive)

`PupilAD-Pack` is a self-contained copy for 64-bit Windows: it carries its own Python 3.12 and
packages, so it runs on any Windows PC without installing anything or going online. It is about
230 MB unzipped. Copy the whole folder, then double-click `Start PupilAD.bat` or
`Start PupilAD for phones and tablets.bat`. Sessions are saved in its `data` folder.

Build or update it from this repository with `pack\build_pack.bat` (needs Python 3.11+ and
internet the first time). By default it writes `..\PupilAD-Pack` next to the repository.

- `pack\build_pack.bat` refreshes the code, and also downloads Python and packages only when they
  are missing or `requirements-app.txt` changed, so routine updates take seconds.
  `--out E:\PupilAD-Pack` builds straight onto a flash drive. The `data` folder is never touched.
- `--rebuild` downloads Python and all packages again.
- `--zip` also writes `PupilAD-Pack.zip`. `--publish` uploads that zip to the "PupilAD Pack
  (latest)" GitHub release, replacing the previous one (needs the GitHub CLI, signed in).

Each build runs a quick smoke test with the pack's own Python. `VERSION.txt` in the pack shows the
code commit it was built from.

## Phones and tablets

`--lan` (used by `start_app_phone.bat`) makes the app listen on the local network over HTTPS,
which browsers require before they allow camera access. It creates a self-signed certificate in
`data/app_certificate/` and prints a link like `https://192.168.1.20:5050/?key=...`. The same
link is shown on the New session page of the computer running the app.

- Open the link on the phone or iPad. The browser warns about the certificate the first time;
  choose "Advanced" then "Proceed" (Chrome) or "Show details" then "visit this website" (Safari).
- The `key` in the link is the only access control. Anyone with the link on the same network can
  start sessions and download data, so share it only with the study team. A new key is made each
  time the app starts.
- The phone and the computer must be on the same Wi-Fi. Some guest or university networks block
  devices from reaching each other.

## Running a session

1. **New session.** Enter the participant ID (letters, digits, `-`, `_`). If the ID already has
   data, tick "Replace the existing files" or choose a new ID. Switch tasks on or off; change
   timings under "Advanced settings". Settings are remembered in the browser and saved with each
   session.
2. **Camera setup.** Tips at the top cover light in front, camera at eye level, sitting close and
   being alone in view. A checklist then guides the participant through five steps, showing one
   instruction at a time:

   | Step | Passes when |
   | --- | --- |
   | Only you in view | exactly one face is detected |
   | Lighting | face brightness 70–215, the background is no more than 45 levels brighter than the face (no backlight), and the two halves of the face differ by under 35% |
   | Camera height | the eyes are between 28% and 56% of the way down the picture (the dotted "eyes" line), with the face roughly centred |
   | Distance | the face fills 28–60% of the picture's width |
   | Eyes clear | eyes are found, have enough contrast, and a pupil is detected |

   **Continue** stays locked until the first four steps pass and the pupil is found in at least
   50% of the last 12 checks (about 8 s) for 3 checks in a row. Change the 50% under Advanced
   settings ("Camera setup: pupil found (%) needed to start"); 0 turns the pupil requirement off.
   The last check's values are saved in `session.json` under `device.setup_check`. The microphone
   level is also shown if Animal Sprint is on.
3. **Session map.** Shows the tasks and the estimated time. Pressing Start enters full screen,
   keeps the screen awake and starts recording.
4. **Each task** opens with an instruction card. The participant can do a short practice round
   (with right/wrong feedback) or go straight to the real round (no feedback). A 5 s settle
   countdown comes before each measured block, and a 15 s break with a breathing guide follows
   it.
5. **Finish.** The page uploads any remaining video, then shows a done screen. The app then
   converts the video and runs pupil extraction and the trial table on its own. Progress shows on
   the Results tab.

To stop early, hold **Hold to stop** for 1.5 s, or press **Shift+Esc** on a keyboard. Finished
tasks and the video recorded so far are kept.

### Answering

| Task | Original name | How to answer |
| --- | --- | --- |
| Letter Match | 2-back | **Match** / **No match** buttons, or Space / N |
| Number Recall | Digit span | On-screen keypad, or number keys, Backspace, Enter |
| Color Clash | Stroop | Four colour buttons, or R / B / G / Y |
| Animal Sprint | Verbal fluency | Speak; the microphone records, and animals are counted afterwards from the audio |

Keyboard hints only appear on devices with a mouse.

### Keeping pupil data clean

During measured trials the screen is a constant mid-grey (#4B5563) with no colour flashes or
feedback, so screen brightness doesn't change pupil size. Coloured backgrounds appear only on
instruction, break and results screens, which are outside the task markers.

### How the pupil is found

For each video frame (`webcam/pupil_detector.py`):

1. Find faces and keep only the **largest**, so someone in the background is ignored.
2. Find up to two eyes in the upper part of that face.
3. In the middle of each eye box, blur slightly and find the darkest spot. If the eye has too
   little contrast (median minus darkest under 12 grey levels), report no pupil.
4. Mark pixels darker than *darkest + 30% of (median − darkest)* and take the connected dark
   region around the darkest spot. Reject it if it's tiny, covers more than 30% of the eye box,
   or is too elongated (over 2.2:1). Its equivalent-circle diameter is the pupil size in pixels.

The threshold adapts to each eye's own brightness. The earlier fixed threshold (grey level 30)
found almost nothing on ordinary webcams. On an ordinary (visible-light) webcam, dark irises and
pupils look similar, so the measured dark region can include part of the iris. Pixel sizes also
depend on distance from the camera, so compare them within a session only.

## Output

Files go to `data/task_battery_logs/` with the same names and columns as the PsychoPy script:

- `{id}_markers.csv`, `{id}_nback.csv`, `{id}_digitspan.csv`, `{id}_stroop.csv`
- `{id}_webcam.mp4` and `{id}_webcam_frame_timestamps.csv`, made from the browser recording
- `{id}_webcam_raw.webm` (or `.mp4` on Safari): the recording exactly as the browser made it
- `{id}_fluency_audio.webm` (or `.mp4`): the Animal Sprint recording
- `{id}_pupil_trace.csv` and `{id}_trial_pupil_table.csv`, made by the automatic analysis
- `{id}_session.json`: settings, device type, screen and camera settings, recording details,
  practice scores, page events (e.g. the participant switched apps) and the pupil detection rate

`{id}_nback.csv` has one extra column, `response` (`match`, `no_match` or empty). `responded` still
means "said match" and no answer still counts as no match, so older scripts read it unchanged.
`rt` is now filled for either answer.

On the Results tab, **Run pupil analysis again** reruns the same code as
`pupilad extract-webcam` and `pupilad webcam-trials`.

## Timing

Stimuli and the recording both use the page's own clock (`performance.now()`), so no clock sync
with the server is needed. Each stimulus onset is stamped at the display frame in which it is
drawn, which carries about one refresh (~17 ms at 60 Hz) of uncertainty.

The video's start time is taken from the browser's recording-start event. Frame times come from the
timestamps inside the recorded file, offset by that start time. Camera latency (typically 30–100 ms,
depending on the device) is not corrected, so video frames may lag the markers by a few frames.
That is small next to the 0.5–2 s pupil response, but it matters for very fine timing. Phone
cameras may also lower their frame rate in dim light; the actual rate is saved in
`session.json` (`webcam.mean_fps`).

## Privacy and consent

The camera records the participant's face and Animal Sprint records their voice. The consent form
must say that video and audio are recorded and where they are stored. In phone mode the recording
is sent over the local network (encrypted with HTTPS) to the computer running the app; nothing is
sent to the internet. The session file does not store the camera's device ID.

## Differences from `run_task_battery.py`

- **N-back answers:** the participant answers every letter with Match or No match, and answers are
  accepted until the next letter appears (2 s), not only while the letter is visible (0.5 s).
  Letters stay on screen for the full 0.5 s and come exactly 2 s apart.
- **Practice rounds** with feedback are optional before each task. Practice trials are not
  written to the task CSVs; their scores are in `session.json`.
- **Pauses:** a 5 s settle countdown before each task, a 1 s pause before each digit sequence and
  a 15 s break between tasks. All can be changed in Advanced settings.
- **Digit span** is answered on a keypad. A "Part 2: reverse order" card comes between the forward
  and backward parts.
- **Verbal fluency** is recorded as audio instead of being typed.
- Tasks can be switched off and parameters changed per session.
- Each task's CSV is saved as soon as that task ends, so an aborted session keeps finished tasks.

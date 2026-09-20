"""
PupilAD 2.0 — Cognitive Task Battery (PsychoPy)
Implements: N-back (2-back), Digit Span (forward + backward),
Stroop, and Semantic Fluency, with event markers sent at each
task's start/end to sync with the pupil-recording device.

-------------------------------------------------------------------
RUNTIME NOTE
-------------------------------------------------------------------
This script requires PsychoPy installed on a machine with an actual
display (not a headless server/Colab) — install via:
    pip install psychopy
or use the Standalone PsychoPy app (recommended for non-technical
setup): https://www.psychopy.org/download.html

If DEVICE_MODE = "webcam" (see below), you also need OpenCV:
    pip install opencv-python

This has NOT been runtime-tested in this environment (no display or
camera available here) — only checked for syntax correctness. Test
it on your actual lab machine before a real session, ideally as a
full dry run with recording on.

-------------------------------------------------------------------
DEVICE_MODE — ONE SWITCH FOR BOTH PHASES OF THE PROJECT
-------------------------------------------------------------------
This script supports two recording setups, controlled by the single
DEVICE_MODE constant below:

  "webcam"       — ISEF / science fair phase. Opens the built-in/USB
                    webcam in a background thread for the whole
                    session, saves an MP4 video, and saves a
                    frame-by-frame timestamp log. Pupil features are
                    then extracted from this video afterward (not
                    live during the task).

  "pupillometer" — Patenting / publication phase, once the clinical
                    pupillometer model is confirmed. No video is
                    captured here; the vendor software/SDK on the
                    pupillometer does its own recording in parallel.
                    This script's only job in that mode is to fire
                    the event markers below.

Both modes write markers via send_marker() using the same PsychoPy
clock (core.getTime()), so webcam frames, pupillometer recordings,
and task events can all be time-aligned after the fact regardless
of which mode a given session used.

-------------------------------------------------------------------
EVENT MARKER WIRING — THE PART THAT DEPENDS ON YOUR DEVICE
-------------------------------------------------------------------
`send_marker()` below currently only writes to a local log file with
a timestamp. In "pupillometer" mode, once your device is confirmed,
replace the body of `send_marker()` with the actual trigger
mechanism, which will be one of:
  - A parallel port / TTL pulse (common: `psychopy.parallel`)
  - A serial port command (common: `pyserial`)
  - A vendor SDK call (e.g., Tobii Pro Lab's External Presenter API,
    which listens for TTL or LSL (Lab Streaming Layer) markers)
  - LSL marker stream (common cross-platform solution — `pylsl`)
In "webcam" mode, the CSV log itself is the sync reference alongside
the frame-timestamp log — no external device markers are needed.

The marker LABELS below already match TASK_EVENTS in ingest_tobii.py
(NBACK_START, NBACK_END, etc.) — don't change these strings unless
you also update ingest_tobii.py's TASK_EVENTS dict to match.
"""

from psychopy import visual, core, event, data, gui
import csv
import random
import string
import threading
from datetime import datetime

# =====================================================================
# CONFIG
# =====================================================================

DEVICE_MODE = "webcam"  # "webcam" (ISEF phase) or "pupillometer" (publication phase)
CAMERA_INDEX = 0         # which webcam to use, if DEVICE_MODE == "webcam"
WEBCAM_TARGET_FPS = 30

SUBJECT_ID = None  # set via dialog at startup
OUTPUT_DIR = "task_battery_logs"

NBACK_N = 2
NBACK_TRIALS = 20
NBACK_LETTER_DURATION = 0.5   # how long each letter is visible on screen
NBACK_SOA = 2.0                # stimulus onset asynchrony: time from one
                                # letter's appearance to the NEXT letter's
                                # appearance. Must be >= NBACK_LETTER_DURATION.

DIGIT_SPAN_START_LENGTH = 3
DIGIT_SPAN_MAX_LENGTH = 9
DIGIT_PRESENT_DURATION = 1.0
DIGIT_ISI = 0.5

STROOP_TRIALS = 30
STROOP_COLORS = ["red", "blue", "green", "yellow"]

SEMANTIC_FLUENCY_DURATION_SEC = 60

INSTRUCTION_TEXT_HEIGHT = 0.04
INSTRUCTION_WRAP_WIDTH = 1.4


# =====================================================================
# INSTRUCTION PAGINATION (fixes text overflowing the window)
# =====================================================================

def show_instructions_paginated(win, pages, height=INSTRUCTION_TEXT_HEIGHT,
                                 wrap_width=INSTRUCTION_WRAP_WIDTH):
    """
    Show a list of instruction 'pages' (strings) one at a time, each
    sized to fit the window, advancing on SPACE. The last page's
    footer says 'begin' instead of 'continue'.
    """
    for i, page_text in enumerate(pages):
        is_last = (i == len(pages) - 1)
        footer = "\n\nPress SPACE to begin." if is_last else "\n\nPress SPACE to continue."
        stim = visual.TextStim(
            win, text=page_text + footer, height=height,
            wrapWidth=wrap_width, alignText="left",
        )
        stim.draw()
        win.flip()
        event.waitKeys(keyList=["space"])


# =====================================================================
# WEBCAM RECORDING (ISEF phase — DEVICE_MODE == "webcam")
# =====================================================================

class WebcamRecorder:
    """
    Records webcam video in a background thread for the full session,
    plus a per-frame timestamp log on the same clock used by
    send_marker(), so frames can be aligned to task events afterward.
    Pupil feature extraction happens offline on the saved video, not
    live during the task.
    """

    def __init__(self, video_path, timestamp_path, camera_index=CAMERA_INDEX,
                 target_fps=WEBCAM_TARGET_FPS):
        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self.camera_index = camera_index
        self.target_fps = target_fps
        self._stop_flag = threading.Event()
        self._thread = None
        self._frame_log = []
        self._error = None

    def start(self):
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        core.wait(0.3)  # brief pause so camera has time to open before tasks start
        if self._error is not None:
            raise RuntimeError(f"Webcam failed to start: {self._error}")

    def _record_loop(self):
        try:
            import cv2
        except ImportError as e:
            self._error = f"OpenCV not installed ({e}). Run: pip install opencv-python"
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self._error = f"Could not open camera index {self.camera_index}"
            return

        writer = None
        frame_idx = 0
        try:
            while not self._stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(self.video_path, fourcc, self.target_fps, (w, h))
                writer.write(frame)
                self._frame_log.append((frame_idx, datetime.now().isoformat(), core.getTime()))
                frame_idx += 1
        finally:
            if writer is not None:
                writer.release()
            cap.release()

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with open(self.timestamp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "wall_clock_time", "psychopy_clock_time"])
            w.writerows(self._frame_log)
        if self._error is not None:
            print(f"[WEBCAM WARNING] {self._error}")


# =====================================================================
# EVENT MARKER LOGGING (placeholder for pupillometer mode — see docstring)
# =====================================================================

def send_marker(win, marker_label: str, log_writer):
    """
    Send an event marker. Always logs to CSV with a timestamp.
    In "pupillometer" mode, once your device is confirmed, ALSO add
    the actual trigger call here (parallel port / serial / SDK / LSL)
    — see module docstring for options. In "webcam" mode, the CSV log
    plus the webcam frame-timestamp log are the full sync reference,
    so no extra device call is needed.
    """
    timestamp = core.getTime()
    log_writer.writerow([datetime.now().isoformat(), timestamp, marker_label])
    print(f"[MARKER] {marker_label} @ {timestamp:.3f}s")

    if DEVICE_MODE == "pupillometer":
        pass
        # --- TODO once device confirmed, e.g. for a parallel port: ---
        # from psychopy import parallel
        # port = parallel.ParallelPort(address=0x0378)
        # port.setData(MARKER_CODES[marker_label])
        #
        # --- or for LSL (common with Tobii Pro Lab External Presenter): ---
        # from pylsl import StreamInfo, StreamOutlet
        # info = StreamInfo('PupilAD_Markers', 'Markers', 1, 0, 'string', 'pupilad001')
        # outlet = StreamOutlet(info)
        # outlet.push_sample([marker_label])


# =====================================================================
# SETUP
# =====================================================================

def get_subject_info():
    info = {"Subject ID": "", "Session": "1"}
    dlg = gui.DlgFromDict(info, title="PupilAD Task Battery")
    if not dlg.OK:
        core.quit()
    return info


def make_log_writer(subject_id):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = f"{OUTPUT_DIR}/{subject_id}_markers.csv"
    f = open(filepath, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["wall_clock_time", "psychopy_clock_time", "event"])
    return f, writer


# =====================================================================
# TASK 1: N-BACK
# =====================================================================

def run_nback(win, log_writer):
    send_marker(win, "NBACK_START", log_writer)

    pages = [
        (
            "N-BACK TASK\n\n"
            f"You will see a series of letters appear one at a time, about every "
            f"{NBACK_SOA:.0f} seconds.\n\n"
            f"Press SPACEBAR whenever the current letter is the SAME as the "
            f"letter that appeared {NBACK_N} letters before it."
        ),
        (
            "N-BACK TASK (example)\n\n"
            "Sequence: K  M  K  T  M  K\n\n"
            "  - 3rd letter (K) vs 1st letter (K) → match → press SPACE\n"
            "  - 4th letter (T) vs 2nd letter (M) → no match → don't press\n"
            "  - 5th letter (M) vs 3rd letter (K) → no match → don't press\n"
            "  - 6th letter (K) vs 4th letter (T) → no match → don't press\n\n"
            "If unsure, it's fine to skip rather than guess. Keep your head "
            "still and eyes on the screen."
        ),
    ]
    show_instructions_paginated(win, pages)

    letters = [random.choice(string.ascii_uppercase) for _ in range(NBACK_TRIALS)]
    # force some real matches so the task isn't trivially all-mismatch
    for i in range(NBACK_N, NBACK_TRIALS):
        if random.random() < 0.3:
            letters[i] = letters[i - NBACK_N]

    results = []
    stim = visual.TextStim(win, text="", height=0.15)

    for i, letter in enumerate(letters):
        stim.text = letter
        stim.draw()
        win.flip()

        rt_clock = core.Clock()
        keys = event.waitKeys(maxWait=NBACK_LETTER_DURATION, keyList=["space"], timeStamped=rt_clock)

        is_match = (i >= NBACK_N) and (letter == letters[i - NBACK_N])
        responded = keys is not None
        rt = keys[0][1] if responded else None

        results.append({
            "trial": i, "letter": letter, "is_match": is_match,
            "responded": responded, "rt": rt,
            "correct": (responded == is_match),
        })

        win.flip()  # blank screen
        core.wait(NBACK_SOA - NBACK_LETTER_DURATION)

    send_marker(win, "NBACK_END", log_writer)
    return results


# =====================================================================
# TASK 2: DIGIT SPAN (forward then backward)
# =====================================================================

def run_digit_span(win, log_writer):
    send_marker(win, "DIGITSPAN_START", log_writer)

    pages = [
        (
            "DIGIT SPAN TASK\n\n"
            "You'll see a sequence of digits, one at a time, each shown briefly "
            "then disappearing before the next appears.\n\n"
            "PART 1 — FORWARD: type the digits back in the SAME order, then "
            "press ENTER.\n\n"
            "PART 2 — BACKWARD: type the digits back in REVERSE order, then "
            "press ENTER."
        ),
        (
            "DIGIT SPAN TASK (example)\n\n"
            "If you saw: 4, 7, 2\n"
            "  - Forward, you would type: 472\n"
            "  - Backward, you would type: 274\n\n"
            "Sequences start at 3 digits and get one digit longer each time you "
            "answer correctly. Each direction ends as soon as you miss one — "
            "just do your best.\n\n"
            "Use BACKSPACE to fix a typo before pressing ENTER."
        ),
    ]
    show_instructions_paginated(win, pages)

    results = []

    for direction in ["forward", "backward"]:
        length = DIGIT_SPAN_START_LENGTH
        while length <= DIGIT_SPAN_MAX_LENGTH:
            digits = [str(random.randint(0, 9)) for _ in range(length)]

            for d in digits:
                stim = visual.TextStim(win, text=d, height=0.2)
                stim.draw()
                win.flip()
                core.wait(DIGIT_PRESENT_DURATION)
                win.flip()
                core.wait(DIGIT_ISI)

            prompt = visual.TextStim(
                win, text="Type the digits you saw, then press ENTER\n"
                          f"({'same order' if direction == 'forward' else 'REVERSE order'})",
                height=0.06,
            )
            prompt.draw()
            win.flip()

            response = ""
            while True:
                keys = event.waitKeys()
                if "return" in keys:
                    break
                elif "backspace" in keys:
                    response = response[:-1]
                else:
                    for k in keys:
                        if k.isdigit():
                            response += k
                display = visual.TextStim(win, text=response, height=0.1)
                display.draw()
                win.flip()

            expected = "".join(digits) if direction == "forward" else "".join(reversed(digits))
            correct = response == expected

            results.append({
                "direction": direction, "length": length,
                "digits": "".join(digits), "response": response, "correct": correct,
            })

            if not correct:
                break  # stop increasing length after first failure, standard digit span protocol
            length += 1

    send_marker(win, "DIGITSPAN_END", log_writer)
    return results


# =====================================================================
# TASK 3: STROOP
# =====================================================================

def run_stroop(win, log_writer):
    send_marker(win, "STROOP_START", log_writer)

    pages = [
        (
            "STROOP TASK\n\n"
            "You'll see color words (RED, BLUE, GREEN, YELLOW) printed in "
            "colored ink. The ink color and the word will often NOT match.\n\n"
            "Respond to the INK COLOR the word is printed in — ignore what the "
            "word says.\n\n"
            "R = red    B = blue    G = green    Y = yellow"
        ),
        (
            "STROOP TASK (example)\n\n"
            "If the word 'GREEN' appears printed in blue ink, press B (for "
            "blue), not G.\n\n"
            "Each trial gives you a few seconds to respond before moving on. "
            "Try to respond as fast and accurately as you can."
        ),
    ]
    show_instructions_paginated(win, pages)

    key_map = {"r": "red", "b": "blue", "g": "green", "y": "yellow"}
    results = []

    for i in range(STROOP_TRIALS):
        word = random.choice(STROOP_COLORS)
        ink_color = random.choice(STROOP_COLORS)
        is_congruent = word == ink_color

        stim = visual.TextStim(win, text=word.upper(), color=ink_color, height=0.15)
        stim.draw()
        win.flip()

        rt_clock = core.Clock()
        keys = event.waitKeys(maxWait=3.0, keyList=list(key_map.keys()), timeStamped=rt_clock)

        responded = keys is not None
        response_color = key_map[keys[0][0]] if responded else None
        rt = keys[0][1] if responded else None

        results.append({
            "trial": i, "word": word, "ink_color": ink_color,
            "is_congruent": is_congruent, "responded": responded,
            "response_color": response_color, "rt": rt,
            "correct": (response_color == ink_color),
        })

        win.flip()
        core.wait(0.5)

    send_marker(win, "STROOP_END", log_writer)
    return results


# =====================================================================
# TASK 4: SEMANTIC FLUENCY
# =====================================================================

def run_semantic_fluency(win, log_writer):
    send_marker(win, "FLUENCY_START", log_writer)

    pages = [
        (
            "SEMANTIC FLUENCY TASK\n\n"
            "You will have 60 seconds to name out loud as many different "
            "ANIMALS as you can think of (e.g. 'dog, cat, elephant, tiger...').\n\n"
            "Say the names out loud; you do not type anything for this task."
        ),
        (
            "SEMANTIC FLUENCY TASK (continued)\n\n"
            "Try not to repeat an animal you've already said. If you get "
            "stuck, keep thinking — any animal counts, including unusual "
            "ones.\n\n"
            "Your responses are recorded separately by the experimenter/"
            "microphone — this screen just shows a countdown timer."
        ),
    ]
    show_instructions_paginated(win, pages)

    timer_clock = core.Clock()
    while timer_clock.getTime() < SEMANTIC_FLUENCY_DURATION_SEC:
        remaining = SEMANTIC_FLUENCY_DURATION_SEC - timer_clock.getTime()
        stim = visual.TextStim(win, text=f"Name animals!\n\n{remaining:.0f}s remaining", height=0.08)
        stim.draw()
        win.flip()
        if event.getKeys(["escape"]):
            break

    send_marker(win, "FLUENCY_END", log_writer)
    # Actual animal count/response scoring happens separately via audio
    # recording + transcription — not captured in this script.
    return []


# =====================================================================
# MAIN
# =====================================================================

def main():
    info = get_subject_info()
    subject_id = info["Subject ID"]

    log_file, log_writer = make_log_writer(subject_id)

    win = visual.Window(fullscr=True, color="grey", units="height")

    webcam_recorder = None
    if DEVICE_MODE == "webcam":
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        webcam_recorder = WebcamRecorder(
            video_path=f"{OUTPUT_DIR}/{subject_id}_webcam.mp4",
            timestamp_path=f"{OUTPUT_DIR}/{subject_id}_webcam_frame_timestamps.csv",
        )
        webcam_recorder.start()

    try:
        nback_results = run_nback(win, log_writer)
        core.wait(2.0)  # brief inter-task rest
        digit_span_results = run_digit_span(win, log_writer)
        core.wait(2.0)
        stroop_results = run_stroop(win, log_writer)
        core.wait(2.0)
        semantic_fluency_results = run_semantic_fluency(win, log_writer)

        end_stim = visual.TextStim(win, text="Session complete. Thank you!", height=0.08)
        end_stim.draw()
        win.flip()
        core.wait(2.0)

        # Save behavioral results (accuracy/RT) separately from pupillometry
        import pandas as pd
        pd.DataFrame(nback_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_nback.csv", index=False)
        pd.DataFrame(digit_span_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_digitspan.csv", index=False)
        pd.DataFrame(stroop_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_stroop.csv", index=False)

    finally:
        if webcam_recorder is not None:
            webcam_recorder.stop()
        log_file.close()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()

#"""
#PupilAD 2.0 — Cognitive Task Battery (PsychoPy)
#Implements: N-back (2-back), Digit Span (forward + backward),
#Stroop, and Semantic Fluency, with event markers sent at each
#task's start/end to sync with the pupillometer recording.
#
#-------------------------------------------------------------------
#RUNTIME NOTE
#-------------------------------------------------------------------
#This script requires PsychoPy installed on a machine with an actual
#display (not a headless server/Colab) — install via:
#    pip install psychopy
#or use the Standalone PsychoPy app (recommended for non-technical
#setup): https://www.psychopy.org/download.html
#
#This has NOT been runtime-tested in this environment (no display
#available here) — only checked for syntax correctness. Test it on
#your actual lab machine before a real session, ideally as a full
#dry run with the pupillometer connected.
#
#-------------------------------------------------------------------
#NO WEBCAM / CAMERA IN THIS SCRIPT — THIS IS INTENTIONAL
#-------------------------------------------------------------------
#This script does not open or display a webcam feed. Since the
#pivot from webcam-based tracking to a clinical-grade pupillometer,
#eye/pupil recording is handled entirely by the pupillometer's own
#vendor software/SDK, running in parallel to this script. This
#script's only job is to present the stimuli and fire event markers
#(see send_marker() below) so the pupillometer recording can later
#be time-synced to specific trial events (e.g. "N-back trial 5
#letter onset"). If you need a live camera preview for participant
#positioning/calibration, that lives in the pupillometer vendor
#software, not here.
#
#-------------------------------------------------------------------
#EVENT MARKER WIRING — THE PART THAT DEPENDS ON YOUR DEVICE
#-------------------------------------------------------------------
#`send_marker()` below currently only writes to a local log file with
#a timestamp. Once your pupillometer is confirmed, replace the body
#of `send_marker()` with the actual trigger mechanism, which will be
#one of:
#  - A parallel port / TTL pulse (common: `psychopy.parallel`)
#  - A serial port command (common: `pyserial`)
#  - A vendor SDK call (e.g., Tobii Pro Lab's External Presenter API,
#    which listens for TTL or LSL (Lab Streaming Layer) markers)
#  - LSL marker stream (common cross-platform solution — `pylsl`)
#
#The marker LABELS below already match TASK_EVENTS in ingest_tobii.py
#(NBACK_START, NBACK_END, etc.) — don't change these strings unless
#you also update ingest_tobii.py's TASK_EVENTS dict to match.
#"""
#
#from psychopy import visual, core, event, data, gui
#import csv
#import random
#import string
#from datetime import datetime
#
## =====================================================================
## CONFIG
## =====================================================================
#
#SUBJECT_ID = None  # set via dialog at startup
#OUTPUT_DIR = "task_battery_logs"
#
#NBACK_N = 2
#NBACK_TRIALS = 20
#NBACK_LETTER_DURATION = 0.5   # how long each letter is visible on screen
#NBACK_SOA = 2.0                # stimulus onset asynchrony: time from one
#                                # letter's appearance to the NEXT letter's
#                                # appearance. Must be >= NBACK_LETTER_DURATION.
#
#DIGIT_SPAN_START_LENGTH = 3
#DIGIT_SPAN_MAX_LENGTH = 9
#DIGIT_PRESENT_DURATION = 1.0
#DIGIT_ISI = 0.5
#
#STROOP_TRIALS = 30
#STROOP_COLORS = ["red", "blue", "green", "yellow"]
#
#SEMANTIC_FLUENCY_DURATION_SEC = 60
#
#
## =====================================================================
## EVENT MARKER LOGGING (placeholder — see module docstring)
## =====================================================================
#
#def send_marker(win, marker_label: str, log_writer):
#    """
#    Send an event marker. Currently logs to CSV with a timestamp only.
#    REPLACE THE BODY HERE with your device's actual trigger call once
#    the pupillometer is confirmed (parallel port / serial / SDK / LSL).
#    """
#    timestamp = core.getTime()
#    log_writer.writerow([datetime.now().isoformat(), timestamp, marker_label])
#    print(f"[MARKER] {marker_label} @ {timestamp:.3f}s")
#
#    # --- TODO once device confirmed, e.g. for a parallel port: ---
#    # from psychopy import parallel
#    # port = parallel.ParallelPort(address=0x0378)
#    # port.setData(MARKER_CODES[marker_label])
#    #
#    # --- or for LSL (common with Tobii Pro Lab External Presenter): ---
#    # from pylsl import StreamInfo, StreamOutlet
#    # info = StreamInfo('PupilAD_Markers', 'Markers', 1, 0, 'string', 'pupilad001')
#    # outlet = StreamOutlet(info)
#    # outlet.push_sample([marker_label])
#
#
## =====================================================================
## SETUP
## =====================================================================
#
#def get_subject_info():
#    info = {"Subject ID": "", "Session": "1"}
#    dlg = gui.DlgFromDict(info, title="PupilAD Task Battery")
#    if not dlg.OK:
#        core.quit()
#    return info
#
#
#def make_log_writer(subject_id):
#    import os
#    os.makedirs(OUTPUT_DIR, exist_ok=True)
#    filepath = f"{OUTPUT_DIR}/{subject_id}_markers.csv"
#    f = open(filepath, "w", newline="")
#    writer = csv.writer(f)
#    writer.writerow(["wall_clock_time", "psychopy_clock_time", "event"])
#    return f, writer
#
#
## =====================================================================
## TASK 1: N-BACK
## =====================================================================
#
#def run_nback(win, log_writer):
#    send_marker(win, "NBACK_START", log_writer)
#
#    instructions = visual.TextStim(
#        win, text=(
#            f"N-BACK TASK\n\n"
#            f"You will see a series of letters appear on the screen one at a time, "
#            f"about every {NBACK_SOA:.0f} seconds.\n\n"
#            f"Your job: press the SPACEBAR whenever the letter currently on screen "
#            f"is the SAME as the letter that appeared {NBACK_N} letters before it "
#            f"(that is, {NBACK_N} steps back in the sequence).\n\n"
#            f"Example sequence: K  M  K  T  M  K\n"
#            f"  - At the 3rd letter (K), compare it to the 1st letter (K) — "
#            f"that's a match, so you would press SPACE.\n"
#            f"  - At the 4th letter (T), compare it to the 2nd letter (M) — "
#            f"no match, so you would NOT press SPACE.\n"
#            f"  - At the 5th letter (M), compare it to the 3rd letter (K) — "
#            f"no match, so you would NOT press SPACE.\n"
#            f"  - At the 6th letter (K), compare it to the 4th letter (T) — "
#            f"no match, so you would NOT press SPACE.\n\n"
#            f"If you're not sure, it's fine to skip pressing SPACE rather than guess. "
#            f"Try to keep your head still and eyes on the screen throughout.\n\n"
#            f"Press SPACE to begin."
#        ), height=0.045,
#    )
#    instructions.draw()
#    win.flip()
#    event.waitKeys(keyList=["space"])
#
#    letters = [random.choice(string.ascii_uppercase) for _ in range(NBACK_TRIALS)]
#    # force some real matches so the task isn't trivially all-mismatch
#    for i in range(NBACK_N, NBACK_TRIALS):
#        if random.random() < 0.3:
#            letters[i] = letters[i - NBACK_N]
#
#    results = []
#    stim = visual.TextStim(win, text="", height=0.15)
#
#    for i, letter in enumerate(letters):
#        stim.text = letter
#        stim.draw()
#        win.flip()
#
#        rt_clock = core.Clock()
#        keys = event.waitKeys(maxWait=NBACK_LETTER_DURATION, keyList=["space"], timeStamped=rt_clock)
#
#        is_match = (i >= NBACK_N) and (letter == letters[i - NBACK_N])
#        responded = keys is not None
#        rt = keys[0][1] if responded else None
#
#        results.append({
#            "trial": i, "letter": letter, "is_match": is_match,
#            "responded": responded, "rt": rt,
#            "correct": (responded == is_match),
#        })
#
#        win.flip()  # blank screen
#        core.wait(NBACK_SOA - NBACK_LETTER_DURATION)
#
#    send_marker(win, "NBACK_END", log_writer)
#    return results
#
#
## =====================================================================
## TASK 2: DIGIT SPAN (forward then backward)
## =====================================================================
#
#def run_digit_span(win, log_writer):
#    send_marker(win, "DIGITSPAN_START", log_writer)
#
#    instructions = visual.TextStim(
#        win, text=(
#            "DIGIT SPAN TASK\n\n"
#            "You will see a sequence of digits, one at a time, each shown briefly "
#            "and then disappearing before the next one appears.\n\n"
#            "PART 1 — FORWARD: After the sequence ends, type the digits back in "
#            "the SAME order you saw them, then press ENTER.\n\n"
#            "PART 2 — BACKWARD: After the sequence ends, type the digits back in "
#            "REVERSE order (last digit shown first), then press ENTER.\n\n"
#            "Example: if you saw 4, 7, 2 — forward you would type 472; "
#            "backward you would type 274.\n\n"
#            "The sequences start short (3 digits) and get one digit longer each "
#            "time you answer correctly. The task ends for that direction as soon "
#            "as you get one sequence wrong, so just do your best on each one.\n\n"
#            "Use BACKSPACE to fix a typo before pressing ENTER.\n\n"
#            "Press SPACE to begin."
#        ), height=0.045,
#    )
#    instructions.draw()
#    win.flip()
#    event.waitKeys(keyList=["space"])
#
#    results = []
#
#    for direction in ["forward", "backward"]:
#        length = DIGIT_SPAN_START_LENGTH
#        while length <= DIGIT_SPAN_MAX_LENGTH:
#            digits = [str(random.randint(0, 9)) for _ in range(length)]
#
#            for d in digits:
#                stim = visual.TextStim(win, text=d, height=0.2)
#                stim.draw()
#                win.flip()
#                core.wait(DIGIT_PRESENT_DURATION)
#                win.flip()
#                core.wait(DIGIT_ISI)
#
#            prompt = visual.TextStim(
#                win, text="Type the digits you saw, then press ENTER\n"
#                          f"({'same order' if direction == 'forward' else 'REVERSE order'})",
#                height=0.06,
#            )
#            prompt.draw()
#            win.flip()
#
#            response = ""
#            while True:
#                keys = event.waitKeys()
#                if "return" in keys:
#                    break
#                elif "backspace" in keys:
#                    response = response[:-1]
#                else:
#                    for k in keys:
#                        if k.isdigit():
#                            response += k
#                display = visual.TextStim(win, text=response, height=0.1)
#                display.draw()
#                win.flip()
#
#            expected = "".join(digits) if direction == "forward" else "".join(reversed(digits))
#            correct = response == expected
#
#            results.append({
#                "direction": direction, "length": length,
#                "digits": "".join(digits), "response": response, "correct": correct,
#            })
#
#            if not correct:
#                break  # stop increasing length after first failure, standard digit span protocol
#            length += 1
#
#    send_marker(win, "DIGITSPAN_END", log_writer)
#    return results
#
#
## =====================================================================
## TASK 3: STROOP
## =====================================================================
#
#def run_stroop(win, log_writer):
#    send_marker(win, "STROOP_START", log_writer)
#
#    instructions = visual.TextStim(
#        win, text=(
#            "STROOP TASK\n\n"
#            "You will see color words (RED, BLUE, GREEN, YELLOW) printed in "
#            "colored ink. The ink color and the word itself will often NOT match "
#            "— for example, the word 'BLUE' might be printed in red ink.\n\n"
#            "Your job: respond to the INK COLOR the word is printed in, and "
#            "IGNORE what the word actually says.\n\n"
#            "Press the key matching the ink color as fast and accurately as you can:\n"
#            "  R = red      B = blue      G = green      Y = yellow\n\n"
#            "Example: if the word 'GREEN' appears printed in blue ink, press B "
#            "(for blue), not G.\n\n"
#            "Each trial gives you a few seconds to respond before moving on. "
#            "Press SPACE to begin."
#        ), height=0.045,
#    )
#    instructions.draw()
#    win.flip()
#    event.waitKeys(keyList=["space"])
#
#    key_map = {"r": "red", "b": "blue", "g": "green", "y": "yellow"}
#    results = []
#
#    for i in range(STROOP_TRIALS):
#        word = random.choice(STROOP_COLORS)
#        ink_color = random.choice(STROOP_COLORS)
#        is_congruent = word == ink_color
#
#        stim = visual.TextStim(win, text=word.upper(), color=ink_color, height=0.15)
#        stim.draw()
#        win.flip()
#
#        rt_clock = core.Clock()
#        keys = event.waitKeys(maxWait=3.0, keyList=list(key_map.keys()), timeStamped=rt_clock)
#
#        responded = keys is not None
#        response_color = key_map[keys[0][0]] if responded else None
#        rt = keys[0][1] if responded else None
#
#        results.append({
#            "trial": i, "word": word, "ink_color": ink_color,
#            "is_congruent": is_congruent, "responded": responded,
#            "response_color": response_color, "rt": rt,
#            "correct": (response_color == ink_color),
#        })
#
#        win.flip()
#        core.wait(0.5)
#
#    send_marker(win, "STROOP_END", log_writer)
#    return results
#
#
## =====================================================================
## TASK 4: SEMANTIC FLUENCY
## =====================================================================
#
#def run_semantic_fluency(win, log_writer):
#    send_marker(win, "FLUENCY_START", log_writer)
#
#    instructions = visual.TextStim(
#        win, text=(
#            "SEMANTIC FLUENCY TASK\n\n"
#            "When the task starts, you will have 60 seconds to name out loud as "
#            "many different ANIMALS as you can think of — for example, 'dog, cat, "
#            "elephant, tiger...' and so on.\n\n"
#            "A few rules:\n"
#            "  - Say the names out loud; you do not type anything for this task.\n"
#            "  - Try not to repeat an animal you've already said.\n"
#            "  - If you get stuck, just keep thinking — any animal counts, "
#            "including ones you might consider unusual.\n"
#            "  - Your responses are recorded separately by the experimenter/"
#            "microphone, not by this screen — this screen just shows you a "
#            "countdown timer.\n\n"
#            "Press SPACE to begin."
#        ), height=0.045,
#    )
#    instructions.draw()
#    win.flip()
#    event.waitKeys(keyList=["space"])
#
#    timer_clock = core.Clock()
#    while timer_clock.getTime() < SEMANTIC_FLUENCY_DURATION_SEC:
#        remaining = SEMANTIC_FLUENCY_DURATION_SEC - timer_clock.getTime()
#        stim = visual.TextStim(win, text=f"Name animals!\n\n{remaining:.0f}s remaining", height=0.08)
#        stim.draw()
#        win.flip()
#        if event.getKeys(["escape"]):
#            break
#
#    send_marker(win, "FLUENCY_END", log_writer)
#    # Actual animal count/response scoring happens separately via audio
#    # recording + transcription — not captured in this script.
#    return []
#
#
## =====================================================================
## MAIN
## =====================================================================
#
#def main():
#    info = get_subject_info()
#    subject_id = info["Subject ID"]
#
#    log_file, log_writer = make_log_writer(subject_id)
#
#    win = visual.Window(fullscr=True, color="grey", units="height")
#
#    try:
#        nback_results = run_nback(win, log_writer)
#        core.wait(2.0)  # brief inter-task rest
#        digit_span_results = run_digit_span(win, log_writer)
#        core.wait(2.0)
#        stroop_results = run_stroop(win, log_writer)
#        core.wait(2.0)
#        semantic_fluency_results = run_semantic_fluency(win, log_writer)
#
#        end_stim = visual.TextStim(win, text="Session complete. Thank you!", height=0.08)
#        end_stim.draw()
#        win.flip()
#        core.wait(2.0)
#
#        # Save behavioral results (accuracy/RT) separately from pupillometry
#        import pandas as pd
#        pd.DataFrame(nback_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_nback.csv", index=False)
#        pd.DataFrame(digit_span_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_digitspan.csv", index=False)
#        pd.DataFrame(stroop_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_stroop.csv", index=False)
#
#    finally:
#        log_file.close()
#        win.close()
#        core.quit()
#
#
#if __name__ == "__main__":
#    main()
#
#
##"""
##PupilAD 2.0 — Cognitive Task Battery (PsychoPy)
##Implements: N-back (2-back), Digit Span (forward + backward),
##Stroop, and Semantic Fluency, with event markers sent at each
##task's start/end to sync with the pupillometer recording.
##
##-------------------------------------------------------------------
##RUNTIME NOTE
##-------------------------------------------------------------------
##This script requires PsychoPy installed on a machine with an actual
##display (not a headless server/Colab) — install via:
##    pip install psychopy
##or use the Standalone PsychoPy app (recommended for non-technical
##setup): https://www.psychopy.org/download.html
##
##This has NOT been runtime-tested in this environment (no display
##available here) — only checked for syntax correctness. Test it on
##your actual lab machine before a real session, ideally as a full
##dry run with the pupillometer connected.
##
##-------------------------------------------------------------------
##EVENT MARKER WIRING — THE PART THAT DEPENDS ON YOUR DEVICE
##-------------------------------------------------------------------
##`send_marker()` below currently only writes to a local log file with
##a timestamp. Once your pupillometer is confirmed, replace the body
##of `send_marker()` with the actual trigger mechanism, which will be
##one of:
##  - A parallel port / TTL pulse (common: `psychopy.parallel`)
##  - A serial port command (common: `pyserial`)
##  - A vendor SDK call (e.g., Tobii Pro Lab's External Presenter API,
##    which listens for TTL or LSL (Lab Streaming Layer) markers)
##  - LSL marker stream (common cross-platform solution — `pylsl`)
##
##The marker LABELS below already match TASK_EVENTS in ingest_tobii.py
##(NBACK_START, NBACK_END, etc.) — don't change these strings unless
##you also update ingest_tobii.py's TASK_EVENTS dict to match.
##"""
##
##from psychopy import visual, core, event, data, gui
##import csv
##import random
##import string
##from datetime import datetime
##
### =====================================================================
### CONFIG
### =====================================================================
##
##SUBJECT_ID = None  # set via dialog at startup
##OUTPUT_DIR = "task_battery_logs"
##
##NBACK_N = 2
##NBACK_TRIALS = 20
##NBACK_LETTER_DURATION = 0.5
##NBACK_ISI = 2.0  # inter-stimulus interval
##
##DIGIT_SPAN_START_LENGTH = 3
##DIGIT_SPAN_MAX_LENGTH = 9
##DIGIT_PRESENT_DURATION = 1.0
##DIGIT_ISI = 0.5
##
##STROOP_TRIALS = 30
##STROOP_COLORS = ["red", "blue", "green", "yellow"]
##
##SEMANTIC_FLUENCY_DURATION_SEC = 60
##
##
### =====================================================================
### EVENT MARKER LOGGING (placeholder — see module docstring)
### =====================================================================
##
##def send_marker(win, marker_label: str, log_writer):
##    """
##    Send an event marker. Currently logs to CSV with a timestamp only.
##    REPLACE THE BODY HERE with your device's actual trigger call once
##    the pupillometer is confirmed (parallel port / serial / SDK / LSL).
##    """
##    timestamp = core.getTime()
##    log_writer.writerow([datetime.now().isoformat(), timestamp, marker_label])
##    print(f"[MARKER] {marker_label} @ {timestamp:.3f}s")
##
##    # --- TODO once device confirmed, e.g. for a parallel port: ---
##    # from psychopy import parallel
##    # port = parallel.ParallelPort(address=0x0378)
##    # port.setData(MARKER_CODES[marker_label])
##    #
##    # --- or for LSL (common with Tobii Pro Lab External Presenter): ---
##    # from pylsl import StreamInfo, StreamOutlet
##    # info = StreamInfo('PupilAD_Markers', 'Markers', 1, 0, 'string', 'pupilad001')
##    # outlet = StreamOutlet(info)
##    # outlet.push_sample([marker_label])
##
##
### =====================================================================
### SETUP
### =====================================================================
##
##def get_subject_info():
##    info = {"Subject ID": "", "Session": "1"}
##    dlg = gui.DlgFromDict(info, title="PupilAD Task Battery")
##    if not dlg.OK:
##        core.quit()
##    return info
##
##
##def make_log_writer(subject_id):
##    import os
##    os.makedirs(OUTPUT_DIR, exist_ok=True)
##    filepath = f"{OUTPUT_DIR}/{subject_id}_markers.csv"
##    f = open(filepath, "w", newline="")
##    writer = csv.writer(f)
##    writer.writerow(["wall_clock_time", "psychopy_clock_time", "event"])
##    return f, writer
##
##
### =====================================================================
### TASK 1: N-BACK
### =====================================================================
##
##def run_nback(win, log_writer):
##    send_marker(win, "NBACK_START", log_writer)
##
##    instructions = visual.TextStim(
##        win, text=(
##            f"N-BACK TASK\n\n"
##            f"Press SPACE if the current letter matches the letter shown "
##            f"{NBACK_N} letters ago.\n\nPress SPACE to begin."
##        ), height=0.06,
##    )
##    instructions.draw()
##    win.flip()
##    event.waitKeys(keyList=["space"])
##
##    letters = [random.choice(string.ascii_uppercase) for _ in range(NBACK_TRIALS)]
##    # force some real matches so the task isn't trivially all-mismatch
##    for i in range(NBACK_N, NBACK_TRIALS):
##        if random.random() < 0.3:
##            letters[i] = letters[i - NBACK_N]
##
##    results = []
##    stim = visual.TextStim(win, text="", height=0.15)
##
##    for i, letter in enumerate(letters):
##        stim.text = letter
##        stim.draw()
##        win.flip()
##
##        rt_clock = core.Clock()
##        keys = event.waitKeys(maxWait=NBACK_LETTER_DURATION, keyList=["space"], timeStamped=rt_clock)
##
##        is_match = (i >= NBACK_N) and (letter == letters[i - NBACK_N])
##        responded = keys is not None
##        rt = keys[0][1] if responded else None
##
##        results.append({
##            "trial": i, "letter": letter, "is_match": is_match,
##            "responded": responded, "rt": rt,
##            "correct": (responded == is_match),
##        })
##
##        win.flip()  # blank screen
##        core.wait(NBACK_ISI - NBACK_LETTER_DURATION)
##
##    send_marker(win, "NBACK_END", log_writer)
##    return results
##
##
### =====================================================================
### TASK 2: DIGIT SPAN (forward then backward)
### =====================================================================
##
##def run_digit_span(win, log_writer):
##    send_marker(win, "DIGITSPAN_START", log_writer)
##
##    instructions = visual.TextStim(
##        win, text=(
##            "DIGIT SPAN TASK\n\n"
##            "You will see a sequence of digits, one at a time.\n"
##            "First, recall them in the SAME order.\n"
##            "Then, a second sequence. Recall it in REVERSE order.\n\n"
##            "Press SPACE to begin."
##        ), height=0.06,
##    )
##    instructions.draw()
##    win.flip()
##    event.waitKeys(keyList=["space"])
##
##    results = []
##
##    for direction in ["forward", "backward"]:
##        length = DIGIT_SPAN_START_LENGTH
##        while length <= DIGIT_SPAN_MAX_LENGTH:
##            digits = [str(random.randint(0, 9)) for _ in range(length)]
##
##            for d in digits:
##                stim = visual.TextStim(win, text=d, height=0.2)
##                stim.draw()
##                win.flip()
##                core.wait(DIGIT_PRESENT_DURATION)
##                win.flip()
##                core.wait(DIGIT_ISI)
##
##            prompt = visual.TextStim(
##                win, text="Type the digits you saw, then press ENTER\n"
##                          f"({'same order' if direction == 'forward' else 'REVERSE order'})",
##                height=0.06,
##            )
##            prompt.draw()
##            win.flip()
##
##            response = ""
##            while True:
##                keys = event.waitKeys()
##                if "return" in keys:
##                    break
##                elif "backspace" in keys:
##                    response = response[:-1]
##                else:
##                    for k in keys:
##                        if k.isdigit():
##                            response += k
##                display = visual.TextStim(win, text=response, height=0.1)
##                display.draw()
##                win.flip()
##
##            expected = "".join(digits) if direction == "forward" else "".join(reversed(digits))
##            correct = response == expected
##
##            results.append({
##                "direction": direction, "length": length,
##                "digits": "".join(digits), "response": response, "correct": correct,
##            })
##
##            if not correct:
##                break  # stop increasing length after first failure, standard digit span protocol
##            length += 1
##
##    send_marker(win, "DIGITSPAN_END", log_writer)
##    return results
##
##
### =====================================================================
### TASK 3: STROOP
### =====================================================================
##
##def run_stroop(win, log_writer):
##    send_marker(win, "STROOP_START", log_writer)
##
##    instructions = visual.TextStim(
##        win, text=(
##            "STROOP TASK\n\n"
##            "Name the INK COLOR of each word, not the word itself.\n"
##            "Press: R = red, B = blue, G = green, Y = yellow\n\n"
##            "Press SPACE to begin."
##        ), height=0.06,
##    )
##    instructions.draw()
##    win.flip()
##    event.waitKeys(keyList=["space"])
##
##    key_map = {"r": "red", "b": "blue", "g": "green", "y": "yellow"}
##    results = []
##
##    for i in range(STROOP_TRIALS):
##        word = random.choice(STROOP_COLORS)
##        ink_color = random.choice(STROOP_COLORS)
##        is_congruent = word == ink_color
##
##        stim = visual.TextStim(win, text=word.upper(), color=ink_color, height=0.15)
##        stim.draw()
##        win.flip()
##
##        rt_clock = core.Clock()
##        keys = event.waitKeys(maxWait=3.0, keyList=list(key_map.keys()), timeStamped=rt_clock)
##
##        responded = keys is not None
##        response_color = key_map[keys[0][0]] if responded else None
##        rt = keys[0][1] if responded else None
##
##        results.append({
##            "trial": i, "word": word, "ink_color": ink_color,
##            "is_congruent": is_congruent, "responded": responded,
##            "response_color": response_color, "rt": rt,
##            "correct": (response_color == ink_color),
##        })
##
##        win.flip()
##        core.wait(0.5)
##
##    send_marker(win, "STROOP_END", log_writer)
##    return results
##
##
### =====================================================================
### TASK 4: SEMANTIC FLUENCY
### =====================================================================
##
##def run_semantic_fluency(win, log_writer):
##    send_marker(win, "FLUENCY_START", log_writer)
##
##    instructions = visual.TextStim(
##        win, text=(
##            "SEMANTIC FLUENCY TASK\n\n"
##            "You will have 60 seconds to name as many ANIMALS as you can, out loud.\n"
##            "(Verbal responses — recorded separately by experimenter/microphone,\n"
##            "not typed here.)\n\n"
##            "Press SPACE to begin."
##        ), height=0.06,
##    )
##    instructions.draw()
##    win.flip()
##    event.waitKeys(keyList=["space"])
##
##    timer_clock = core.Clock()
##    while timer_clock.getTime() < SEMANTIC_FLUENCY_DURATION_SEC:
##        remaining = SEMANTIC_FLUENCY_DURATION_SEC - timer_clock.getTime()
##        stim = visual.TextStim(win, text=f"Name animals!\n\n{remaining:.0f}s remaining", height=0.08)
##        stim.draw()
##        win.flip()
##        if event.getKeys(["escape"]):
##            break
##
##    send_marker(win, "FLUENCY_END", log_writer)
##    # Actual animal count/response scoring happens separately via audio
##    # recording + transcription — not captured in this script.
##    return []
##
##
### =====================================================================
### MAIN
### =====================================================================
##
##def main():
##    info = get_subject_info()
##    subject_id = info["Subject ID"]
##
##    log_file, log_writer = make_log_writer(subject_id)
##
##    win = visual.Window(fullscr=True, color="grey", units="height")
##
##    try:
##        nback_results = run_nback(win, log_writer)
##        core.wait(2.0)  # brief inter-task rest
##        digit_span_results = run_digit_span(win, log_writer)
##        core.wait(2.0)
##        stroop_results = run_stroop(win, log_writer)
##        core.wait(2.0)
##        semantic_fluency_results = run_semantic_fluency(win, log_writer)
##
##        end_stim = visual.TextStim(win, text="Session complete. Thank you!", height=0.08)
##        end_stim.draw()
##        win.flip()
##        core.wait(2.0)
##
##        # Save behavioral results (accuracy/RT) separately from pupillometry
##        import pandas as pd
##        pd.DataFrame(nback_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_nback.csv", index=False)
##        pd.DataFrame(digit_span_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_digitspan.csv", index=False)
##        pd.DataFrame(stroop_results).to_csv(f"{OUTPUT_DIR}/{subject_id}_stroop.csv", index=False)
##
##    finally:
##        log_file.close()
##        win.close()
##        core.quit()
##
##
##if __name__ == "__main__":
##    main()
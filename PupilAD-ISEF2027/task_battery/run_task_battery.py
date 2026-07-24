"""
PupilAD 2.0 — Cognitive Task Battery (PsychoPy)
Implements: N-back (2-back), Digit Span (forward + backward),
Stroop, and Semantic Fluency, with event markers sent at each
task's start/end to sync with the pupillometer recording.

-------------------------------------------------------------------
RUNTIME NOTE
-------------------------------------------------------------------
This script requires PsychoPy installed on a machine with an actual
display (not a headless server/Colab) — install via:
    pip install psychopy
or use the Standalone PsychoPy app (recommended for non-technical
setup): https://www.psychopy.org/download.html

This has NOT been runtime-tested in this environment (no display
available here) — only checked for syntax correctness. Test it on
your actual lab machine before a real session, ideally as a full
dry run with the pupillometer connected.

-------------------------------------------------------------------
EVENT MARKER WIRING — THE PART THAT DEPENDS ON YOUR DEVICE
-------------------------------------------------------------------
`send_marker()` below currently only writes to a local log file with
a timestamp. Once your pupillometer is confirmed, replace the body
of `send_marker()` with the actual trigger mechanism, which will be
one of:
  - A parallel port / TTL pulse (common: `psychopy.parallel`)
  - A serial port command (common: `pyserial`)
  - A vendor SDK call (e.g., Tobii Pro Lab's External Presenter API,
    which listens for TTL or LSL (Lab Streaming Layer) markers)
  - LSL marker stream (common cross-platform solution — `pylsl`)

The marker LABELS below already match TASK_EVENTS in ingest_tobii.py
(NBACK_START, NBACK_END, etc.) — don't change these strings unless
you also update ingest_tobii.py's TASK_EVENTS dict to match.
"""

from psychopy import visual, core, event, data, gui
import csv
import random
import string
from datetime import datetime

# =====================================================================
# CONFIG
# =====================================================================

SUBJECT_ID = None  # set via dialog at startup
OUTPUT_DIR = "task_battery_logs"

NBACK_N = 2
NBACK_TRIALS = 20
NBACK_LETTER_DURATION = 0.5
NBACK_ISI = 2.0  # inter-stimulus interval

DIGIT_SPAN_START_LENGTH = 3
DIGIT_SPAN_MAX_LENGTH = 9
DIGIT_PRESENT_DURATION = 1.0
DIGIT_ISI = 0.5

STROOP_TRIALS = 30
STROOP_COLORS = ["red", "blue", "green", "yellow"]

SEMANTIC_FLUENCY_DURATION_SEC = 60


# =====================================================================
# EVENT MARKER LOGGING (placeholder — see module docstring)
# =====================================================================

def send_marker(win, marker_label: str, log_writer):
    """
    Send an event marker. Currently logs to CSV with a timestamp only.
    REPLACE THE BODY HERE with your device's actual trigger call once
    the pupillometer is confirmed (parallel port / serial / SDK / LSL).
    """
    timestamp = core.getTime()
    log_writer.writerow([datetime.now().isoformat(), timestamp, marker_label])
    print(f"[MARKER] {marker_label} @ {timestamp:.3f}s")

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

    instructions = visual.TextStim(
        win, text=(
            f"N-BACK TASK\n\n"
            f"Press SPACE if the current letter matches the letter shown "
            f"{NBACK_N} letters ago.\n\nPress SPACE to begin."
        ), height=0.06,
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

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
        core.wait(NBACK_ISI - NBACK_LETTER_DURATION)

    send_marker(win, "NBACK_END", log_writer)
    return results


# =====================================================================
# TASK 2: DIGIT SPAN (forward then backward)
# =====================================================================

def run_digit_span(win, log_writer):
    send_marker(win, "DIGITSPAN_START", log_writer)

    instructions = visual.TextStim(
        win, text=(
            "DIGIT SPAN TASK\n\n"
            "You will see a sequence of digits, one at a time.\n"
            "First, recall them in the SAME order.\n"
            "Then, a second sequence — recall it in REVERSE order.\n\n"
            "Press SPACE to begin."
        ), height=0.06,
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

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

    instructions = visual.TextStim(
        win, text=(
            "STROOP TASK\n\n"
            "Name the INK COLOR of each word, not the word itself.\n"
            "Press: R = red, B = blue, G = green, Y = yellow\n\n"
            "Press SPACE to begin."
        ), height=0.06,
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

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

    instructions = visual.TextStim(
        win, text=(
            "SEMANTIC FLUENCY TASK\n\n"
            "You will have 60 seconds to name as many ANIMALS as you can, out loud.\n"
            "(Verbal responses — recorded separately by experimenter/microphone,\n"
            "not typed here.)\n\n"
            "Press SPACE to begin."
        ), height=0.06,
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

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
        log_file.close()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()

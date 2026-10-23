import csv
import random
import string
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
from psychopy import core, event, gui, visual

RECORD_WEBCAM = True
CAMERA_INDEX = 0
WEBCAM_TARGET_FPS = 30
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "task_battery_logs"

NBACK_N = 2
NBACK_TRIALS = 20
NBACK_LETTER_DURATION_S = 0.5
NBACK_SOA_S = 2.0
NBACK_MATCH_PROBABILITY = 0.3

DIGIT_SPAN_START_LENGTH = 3
DIGIT_SPAN_MAX_LENGTH = 9
DIGIT_PRESENT_DURATION_S = 1.0
DIGIT_ISI_S = 0.5

STROOP_TRIALS = 30
STROOP_COLORS = ["red", "blue", "green", "yellow"]
STROOP_KEYS = {"r": "red", "b": "blue", "g": "green", "y": "yellow"}
STROOP_RESPONSE_WINDOW_S = 3.0
STROOP_ISI_S = 0.5

FLUENCY_DURATION_S = 60

INSTRUCTION_TEXT_HEIGHT = 0.04
TEXT_WRAP_WIDTH = 1.4
REMINDER_HEIGHT = 0.035
REMINDER_POS = (0, -0.42)
PROGRESS_POS = (0, 0.44)
REMINDER_COLOR = "grey"


def show_instruction_pages(win, pages):
    for index, page_text in enumerate(pages):
        footer = "begin" if index == len(pages) - 1 else "continue"
        visual.TextStim(
            win,
            text=f"{page_text}\n\nPress SPACE to {footer}.",
            height=INSTRUCTION_TEXT_HEIGHT,
            wrapWidth=TEXT_WRAP_WIDTH,
            alignText="left",
        ).draw()
        win.flip()
        event.waitKeys(keyList=["space"])


def reminder_text(win, text):
    return visual.TextStim(
        win,
        text=text,
        height=REMINDER_HEIGHT,
        pos=REMINDER_POS,
        color=REMINDER_COLOR,
        wrapWidth=TEXT_WRAP_WIDTH,
    )


def progress_text(win):
    return visual.TextStim(
        win, text="", height=REMINDER_HEIGHT, pos=PROGRESS_POS, color=REMINDER_COLOR
    )


def show_task_transition(win, task_number, total_tasks, task_name, pause_s=2.0):
    visual.TextStim(
        win,
        text=f"Task {task_number} of {total_tasks}\n\n{task_name}",
        height=0.07,
        wrapWidth=TEXT_WRAP_WIDTH,
    ).draw()
    win.flip()
    core.wait(pause_s)


def show_countdown(win, seconds=3):
    for remaining in range(seconds, 0, -1):
        visual.TextStim(win, text=f"Get ready...\n\n{remaining}", height=0.1, color="yellow").draw()
        win.flip()
        core.wait(1.0)


def draw_all(win, *stimuli):
    for stimulus in stimuli:
        stimulus.draw()
    win.flip()


class WebcamRecorder:
    def __init__(self, video_path, timestamp_path):
        self.video_path = str(video_path)
        self.timestamp_path = timestamp_path
        self.stop_requested = threading.Event()
        self.thread = None
        self.frame_log = []
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self.record, daemon=True)
        self.thread.start()
        core.wait(0.3)
        if self.error is not None:
            raise RuntimeError(f"Webcam failed to start: {self.error}")

    def record(self):
        try:
            import cv2
        except ImportError as error:
            self.error = f"OpenCV not installed ({error})"
            return
        capture = cv2.VideoCapture(CAMERA_INDEX)
        if not capture.isOpened():
            self.error = f"Could not open camera index {CAMERA_INDEX}"
            return
        writer = None
        try:
            while not self.stop_requested.is_set():
                ok, frame = capture.read()
                if not ok:
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        self.video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        WEBCAM_TARGET_FPS,
                        (width, height),
                    )
                writer.write(frame)
                self.frame_log.append(
                    (len(self.frame_log), datetime.now().isoformat(), core.getTime())
                )
        finally:
            if writer is not None:
                writer.release()
            capture.release()

    def stop(self):
        self.stop_requested.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        with open(self.timestamp_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "wall_clock_time", "psychopy_clock_time"])
            writer.writerows(self.frame_log)
        if self.error is not None:
            print(f"Webcam warning: {self.error}")


def send_marker(log_writer, label):
    timestamp = core.getTime()
    log_writer.writerow([datetime.now().isoformat(), timestamp, label])


def ask_subject_id():
    info = {"Subject ID": "", "Session": "1"}
    if not gui.DlgFromDict(info, title="PupilAD Task Battery").OK:
        core.quit()
    return info["Subject ID"]


def open_marker_log(subject_id):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(OUTPUT_DIR / f"{subject_id}_markers.csv", "w", newline="")
    writer = csv.writer(handle)
    writer.writerow(["wall_clock_time", "psychopy_clock_time", "event"])
    return handle, writer


NBACK_PAGES = [
    "N-BACK TASK\n\n"
    f"You will see a series of letters appear one at a time, about every {NBACK_SOA_S:.0f} seconds.\n\n"
    f"Press SPACEBAR whenever the current letter is the SAME as the letter that appeared {NBACK_N} letters before it.",
    "N-BACK TASK (example)\n\n"
    "Sequence: K  M  K  T  M  K\n\n"
    "  - 3rd letter (K) vs 1st letter (K) → match → press SPACE\n"
    "  - 4th letter (T) vs 2nd letter (M) → no match → don't press\n"
    "  - 5th letter (M) vs 3rd letter (K) → no match → don't press\n"
    "  - 6th letter (K) vs 4th letter (T) → no match → don't press\n\n"
    "If unsure, it's fine to skip rather than guess. Keep your head still and eyes on the screen.",
]

DIGIT_SPAN_PAGES = [
    "DIGIT SPAN TASK\n\n"
    "You'll see a sequence of digits, one at a time, each shown briefly then disappearing before the next appears.\n\n"
    "PART 1 — FORWARD: type the digits back in the SAME order, then press ENTER.\n\n"
    "PART 2 — BACKWARD: type the digits back in REVERSE order, then press ENTER.",
    "DIGIT SPAN TASK (example)\n\n"
    "If you saw: 4, 7, 2\n"
    "  - Forward, you would type: 472\n"
    "  - Backward, you would type: 274\n\n"
    "Sequences start at 3 digits and get one digit longer each time you answer correctly. "
    "Each direction ends as soon as you miss one — just do your best.\n\n"
    "Use BACKSPACE to fix a typo before pressing ENTER.",
]

STROOP_PAGES = [
    "STROOP TASK\n\n"
    "You'll see color words (RED, BLUE, GREEN, YELLOW) printed in colored ink. "
    "The ink color and the word will often NOT match.\n\n"
    "Respond to the INK COLOR the word is printed in — ignore what the word says.\n\n"
    "R = red    B = blue    G = green    Y = yellow",
    "STROOP TASK (example)\n\n"
    "If the word 'GREEN' appears printed in blue ink, press B (for blue), not G.\n\n"
    "Each trial gives you a few seconds to respond before moving on. "
    "Try to respond as fast and accurately as you can.",
]

FLUENCY_PAGES = [
    "SEMANTIC FLUENCY TASK\n\n"
    "You will have 60 seconds to name out loud as many different ANIMALS as you can think of "
    "(e.g. 'dog, cat, elephant, tiger...').\n\n"
    "Say the names out loud; you do not type anything for this task.",
    "SEMANTIC FLUENCY TASK (continued)\n\n"
    "Try not to repeat an animal you've already said. If you get stuck, keep thinking — "
    "any animal counts, including unusual ones.\n\n"
    "Your responses are recorded separately by the experimenter/microphone — "
    "this screen just shows a countdown timer.",
]


def nback_letters():
    letters = [random.choice(string.ascii_uppercase) for _ in range(NBACK_TRIALS)]
    for index in range(NBACK_N, NBACK_TRIALS):
        if random.random() < NBACK_MATCH_PROBABILITY:
            letters[index] = letters[index - NBACK_N]
    return letters


def run_nback(win, log_writer):
    show_instruction_pages(win, NBACK_PAGES)
    show_countdown(win)
    send_marker(log_writer, "NBACK_START")
    letters = nback_letters()
    stimulus = visual.TextStim(win, text="", height=0.15)
    reminder = reminder_text(
        win, f"Press SPACE if this letter matches the one from {NBACK_N} letters back"
    )
    progress = progress_text(win)
    results = []
    for index, letter in enumerate(letters):
        stimulus.text = letter
        progress.text = f"Trial {index + 1} of {NBACK_TRIALS}"
        draw_all(win, stimulus, reminder, progress)
        onset_time = core.getTime()
        keys = event.waitKeys(
            maxWait=NBACK_LETTER_DURATION_S, keyList=["space"], timeStamped=core.Clock()
        )
        is_match = index >= NBACK_N and letter == letters[index - NBACK_N]
        responded = keys is not None
        results.append(
            {
                "trial": index,
                "letter": letter,
                "is_match": is_match,
                "responded": responded,
                "rt": keys[0][1] if responded else None,
                "correct": responded == is_match,
                "onset_time": onset_time,
            }
        )
        draw_all(win, reminder, progress)
        core.wait(NBACK_SOA_S - NBACK_LETTER_DURATION_S)
    send_marker(log_writer, "NBACK_END")
    return results


def present_digits(win, digits, progress):
    for digit in digits:
        draw_all(win, visual.TextStim(win, text=digit, height=0.2), progress)
        core.wait(DIGIT_PRESENT_DURATION_S)
        draw_all(win, progress)
        core.wait(DIGIT_ISI_S)


def collect_typed_digits(win, prompt, progress):
    response = ""
    draw_all(win, prompt, progress)
    while True:
        keys = event.waitKeys()
        if "return" in keys:
            return response, core.getTime()
        if "backspace" in keys:
            response = response[:-1]
        else:
            response += "".join(key for key in keys if key.isdigit())
        draw_all(win, prompt, visual.TextStim(win, text=response, height=0.1), progress)


def run_digit_span(win, log_writer):
    show_instruction_pages(win, DIGIT_SPAN_PAGES)
    show_countdown(win)
    send_marker(log_writer, "DIGITSPAN_START")
    progress = progress_text(win)
    results = []
    for direction in ("forward", "backward"):
        order_hint = "same order" if direction == "forward" else "REVERSE order"
        prompt = visual.TextStim(
            win,
            text=f"Type the digits you saw, then press ENTER\n({order_hint})",
            height=0.06,
            pos=(0, 0.15),
        )
        for length in range(DIGIT_SPAN_START_LENGTH, DIGIT_SPAN_MAX_LENGTH + 1):
            digits = [str(random.randint(0, 9)) for _ in range(length)]
            progress.text = f"{direction.upper()} — sequence length {length}"
            stim_onset_time = core.getTime()
            present_digits(win, digits, progress)
            response, recall_end_time = collect_typed_digits(win, prompt, progress)
            expected = "".join(digits if direction == "forward" else reversed(digits))
            correct = response == expected
            results.append(
                {
                    "direction": direction,
                    "length": length,
                    "digits": "".join(digits),
                    "response": response,
                    "correct": correct,
                    "stim_onset_time": stim_onset_time,
                    "recall_end_time": recall_end_time,
                }
            )
            if not correct:
                break
    send_marker(log_writer, "DIGITSPAN_END")
    return results


def run_stroop(win, log_writer):
    show_instruction_pages(win, STROOP_PAGES)
    show_countdown(win)
    send_marker(log_writer, "STROOP_START")
    reminder = reminder_text(win, "R = red      B = blue      G = green      Y = yellow")
    progress = progress_text(win)
    results = []
    for index in range(STROOP_TRIALS):
        word = random.choice(STROOP_COLORS)
        ink_color = random.choice(STROOP_COLORS)
        progress.text = f"Trial {index + 1} of {STROOP_TRIALS}"
        draw_all(
            win,
            visual.TextStim(win, text=word.upper(), color=ink_color, height=0.15),
            reminder,
            progress,
        )
        onset_time = core.getTime()
        keys = event.waitKeys(
            maxWait=STROOP_RESPONSE_WINDOW_S, keyList=list(STROOP_KEYS), timeStamped=core.Clock()
        )
        responded = keys is not None
        response_color = STROOP_KEYS[keys[0][0]] if responded else None
        results.append(
            {
                "trial": index,
                "word": word,
                "ink_color": ink_color,
                "is_congruent": word == ink_color,
                "responded": responded,
                "response_color": response_color,
                "rt": keys[0][1] if responded else None,
                "correct": response_color == ink_color,
                "onset_time": onset_time,
            }
        )
        draw_all(win, reminder, progress)
        core.wait(STROOP_ISI_S)
    send_marker(log_writer, "STROOP_END")
    return results


def run_semantic_fluency(win, log_writer):
    show_instruction_pages(win, FLUENCY_PAGES)
    show_countdown(win)
    send_marker(log_writer, "FLUENCY_START")
    timer = core.Clock()
    while timer.getTime() < FLUENCY_DURATION_S:
        remaining = FLUENCY_DURATION_S - timer.getTime()
        draw_all(
            win,
            visual.TextStim(win, text=f"Name animals!\n\n{remaining:.0f}s remaining", height=0.08),
        )
        if event.getKeys(["escape"]):
            break
    send_marker(log_writer, "FLUENCY_END")


def save_results(subject_id, results_by_task):
    for task_name, results in results_by_task.items():
        pd.DataFrame(results).to_csv(OUTPUT_DIR / f"{subject_id}_{task_name}.csv", index=False)


def run_battery(win, log_writer):
    show_task_transition(win, 1, 4, "N-Back")
    nback = run_nback(win, log_writer)
    show_task_transition(win, 2, 4, "Digit Span")
    digit_span = run_digit_span(win, log_writer)
    show_task_transition(win, 3, 4, "Stroop")
    stroop = run_stroop(win, log_writer)
    show_task_transition(win, 4, 4, "Semantic Fluency")
    run_semantic_fluency(win, log_writer)
    draw_all(win, visual.TextStim(win, text="Session complete. Thank you!", height=0.08))
    core.wait(2.0)
    return {"nback": nback, "digitspan": digit_span, "stroop": stroop}


def main():
    subject_id = ask_subject_id()
    log_handle, log_writer = open_marker_log(subject_id)
    win = visual.Window(fullscr=True, color="grey", units="height")
    recorder = None
    if RECORD_WEBCAM:
        recorder = WebcamRecorder(
            OUTPUT_DIR / f"{subject_id}_webcam.mp4",
            OUTPUT_DIR / f"{subject_id}_webcam_frame_timestamps.csv",
        )
        recorder.start()
    try:
        save_results(subject_id, run_battery(win, log_writer))
    finally:
        if recorder is not None:
            recorder.stop()
        log_handle.close()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()

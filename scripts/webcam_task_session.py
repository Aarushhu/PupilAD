import random
import time

import cv2
import numpy as np
import pandas as pd

from pupilad.paths import DATA_DIR
from pupilad.webcam.pupil_detector import detect_pupil_diameter_px

TASK_DURATION_S = 90
LEAD_IN_S = 2
PANEL_SHAPE = (400, 700, 3)
PREVIEW_SIZE = (233, 175)
PREVIEW_SLOT = (slice(210, 385), slice(460, 693))
OUTPUT_DIR = DATA_DIR / "webcam_pilot"
FONT = cv2.FONT_HERSHEY_SIMPLEX
QUIT_KEY = ord("q")

NBACK_LETTERS = "BCDFGHJKLMNPQRSTVWXYZ"
NBACK_LENGTH = 60
NBACK_INTERVAL_S = 2.0
DIGIT_INTERVAL_S = 1.0
RECALL_DURATION_S = 5.0
STROOP_INTERVAL_S = 3.0
FLUENCY_CATEGORY_INTERVAL_S = 20.0

STROOP_ITEMS = [
    ("RED", (0, 0, 255)),
    ("BLUE", (255, 0, 0)),
    ("GREEN", (0, 200, 0)),
    ("RED", (255, 0, 0)),
    ("BLUE", (0, 0, 255)),
    ("GREEN", (0, 255, 128)),
    ("RED", (0, 200, 0)),
    ("BLUE", (0, 200, 0)),
    ("GREEN", (0, 0, 255)),
    ("RED", (255, 0, 0)),
]
FLUENCY_CATEGORIES = {
    "Animals": ["dog", "cat", "elephant"],
    "Fruits": ["apple", "mango", "grape"],
    "Countries": ["Brazil", "Japan", "Kenya"],
}


def nback_sequence(length=NBACK_LENGTH):
    letters = [random.choice(NBACK_LETTERS) for _ in range(length)]
    return [
        (letter, index >= 2 and letter == letters[index - 2])
        for index, letter in enumerate(letters)
    ]


def digit_sequence():
    return [str(random.randint(1, 9)) for _ in range(random.randint(5, 9))]


def blank_panel(title, instruction, seconds_left):
    panel = np.zeros(PANEL_SHAPE, dtype=np.uint8)
    cv2.putText(panel, title, (200, 40), FONT, 1.0, (200, 200, 200), 2)
    cv2.putText(panel, instruction, (60, 75), FONT, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, f"Time: {int(seconds_left)}s", (10, 390), FONT, 0.6, (100, 100, 100), 1)
    return panel


def show(panel, frame):
    panel[PREVIEW_SLOT] = cv2.resize(frame, PREVIEW_SIZE)
    cv2.imshow("PupilAD Task", panel)
    return cv2.waitKey(30) & 0xFF == QUIT_KEY


class NBackScreen:
    title = "2-BACK TASK"
    instruction = "Press SPACE if letter matches 2 back"

    def __init__(self):
        self.sequence = nback_sequence()
        self.index = 0
        self.last_switch = time.time()

    def draw(self, panel):
        if time.time() - self.last_switch > NBACK_INTERVAL_S and self.index < len(self.sequence):
            self.index += 1
            self.last_switch = time.time()
        if self.index < len(self.sequence):
            letter, is_target = self.sequence[self.index]
            color = (0, 255, 0) if is_target else (255, 255, 255)
            cv2.putText(panel, letter, (310, 240), FONT, 6, color, 8)


class DigitSpanScreen:
    title = "DIGIT SPAN"
    instruction = "Remember the sequence - recall when blank"

    def __init__(self):
        self.sequence = digit_sequence()
        self.index = 0
        self.recalling = False
        self.phase_start = time.time()

    def draw(self, panel):
        phase_elapsed = time.time() - self.phase_start
        if self.recalling:
            cv2.putText(panel, "RECALL", (255, 200), FONT, 2, (0, 200, 255), 3)
            cv2.putText(panel, " ".join(self.sequence), (150, 270), FONT, 1.2, (100, 100, 100), 2)
            if phase_elapsed > RECALL_DURATION_S:
                self.__init__()
            return
        if phase_elapsed > DIGIT_INTERVAL_S:
            self.index += 1
            self.phase_start = time.time()
            if self.index >= len(self.sequence):
                self.recalling = True
                return
        cv2.putText(panel, self.sequence[self.index], (310, 240), FONT, 6, (255, 255, 255), 8)


class StroopScreen:
    title = "STROOP TASK"
    instruction = "Say the COLOR of the ink, not the word"

    def __init__(self):
        self.index = 0
        self.last_switch = time.time()

    def draw(self, panel):
        if time.time() - self.last_switch > STROOP_INTERVAL_S:
            self.index = (self.index + 1) % len(STROOP_ITEMS)
            self.last_switch = time.time()
        word, color = STROOP_ITEMS[self.index]
        cv2.putText(panel, word, (160, 240), FONT, 4, color, 8)


class FluencyScreen:
    title = "SEMANTIC FLUENCY"
    instruction = "Say as many items as you can in the category"

    def __init__(self):
        self.categories = list(FLUENCY_CATEGORIES)
        self.index = 0
        self.last_switch = time.time()

    def draw(self, panel):
        if time.time() - self.last_switch > FLUENCY_CATEGORY_INTERVAL_S:
            self.index = (self.index + 1) % len(self.categories)
            self.last_switch = time.time()
        category = self.categories[self.index]
        examples = ", ".join(FLUENCY_CATEGORIES[category]) + "..."
        cv2.putText(panel, category.upper(), (230, 200), FONT, 3, (0, 220, 255), 6)
        cv2.putText(panel, f"e.g. {examples}", (100, 270), FONT, 0.6, (120, 120, 120), 1)


def run_task(capture, task_name, screen, duration_s=TASK_DURATION_S):
    samples = []
    start = time.time() + LEAD_IN_S
    while time.time() - start < duration_s:
        ok, frame = capture.read()
        if not ok:
            break
        elapsed = time.time() - start
        panel = blank_panel(screen.title, screen.instruction, duration_s - elapsed)
        screen.draw(panel)
        diameter_px = detect_pupil_diameter_px(frame)
        if diameter_px:
            samples.append({"timestamp": elapsed, "avg_diameter": diameter_px, "task": task_name})
        if show(panel, frame):
            break
    return pd.DataFrame(samples)


def run_session():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("N-back", NBackScreen),
        ("Digit Span", DigitSpanScreen),
        ("Stroop", StroopScreen),
        ("Semantic Fluency", FluencyScreen),
    ]
    input("4 tasks x 90 seconds. Press ENTER to begin...")
    for task_name, screen_type in tasks:
        input(f"Press ENTER to start: {task_name}")
        samples = run_task(capture, task_name, screen_type())
        path = OUTPUT_DIR / f"pupil_{task_name.replace(' ', '_')}.csv"
        samples.to_csv(path, index=False)
        print(f"Saved {len(samples)} frames to {path}")
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_session()

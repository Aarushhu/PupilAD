import cv2
import numpy as np

FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (100, 100)
EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 5
DARK_PUPIL_THRESHOLD = 30
MIN_PUPIL_AREA_PX = 20

_face_cascade = None
_eye_cascade = None


def cascades():
    global _face_cascade, _eye_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        _eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return _face_cascade, _eye_cascade


def equivalent_diameter(area):
    return 2 * np.sqrt(area / np.pi)


def largest_dark_blob_area(eye_region):
    _, dark = cv2.threshold(eye_region, DARK_PUPIL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    return cv2.contourArea(max(contours, key=cv2.contourArea))


def eye_regions(gray_frame):
    face_cascade, eye_cascade = cascades()
    faces = face_cascade.detectMultiScale(
        gray_frame, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, minSize=FACE_MIN_SIZE
    )
    for face_x, face_y, face_w, face_h in faces:
        face = gray_frame[face_y : face_y + face_h, face_x : face_x + face_w]
        for eye_x, eye_y, eye_w, eye_h in eye_cascade.detectMultiScale(
            face, EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS
        ):
            yield face[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w]


def detect_pupil_diameter_px(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diameters = [
        equivalent_diameter(area)
        for area in (largest_dark_blob_area(region) for region in eye_regions(gray))
        if area > MIN_PUPIL_AREA_PX
    ]
    return float(np.mean(diameters)) if diameters else None

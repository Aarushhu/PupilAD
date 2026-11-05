import cv2
import numpy as np

from pupilad.webcam.pupil_detector import (
    EYE_MIN_NEIGHBORS,
    EYE_SCALE_FACTOR,
    FACE_MIN_NEIGHBORS,
    FACE_SCALE_FACTOR,
    MIN_PUPIL_AREA_PX,
    cascades,
    equivalent_diameter,
    largest_dark_blob_area,
)

MIN_FACE_FRACTION = 0.12


def inspect_frame(jpeg_bytes):
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the image")
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade, eye_cascade = cascades()
    min_side = max(40, int(min(width, height) * MIN_FACE_FRACTION))
    faces = face_cascade.detectMultiScale(
        gray, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, minSize=(min_side, min_side)
    )
    result = {"width": width, "height": height, "face": None, "eyes": 0, "pupil_px": None}
    if len(faces) == 0:
        return result
    face_x, face_y, face_w, face_h = max(faces, key=lambda box: box[2] * box[3])
    result["face"] = {
        "x": face_x / width,
        "y": face_y / height,
        "w": face_w / width,
        "h": face_h / height,
    }
    face = gray[face_y : face_y + face_h, face_x : face_x + face_w]
    eyes = eye_cascade.detectMultiScale(face, EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS)
    eyes = [box for box in eyes if box[1] < face_h * 0.6]
    result["eyes"] = len(eyes)
    diameters = []
    for eye_x, eye_y, eye_w, eye_h in eyes:
        area = largest_dark_blob_area(face[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w])
        if area > MIN_PUPIL_AREA_PX:
            diameters.append(equivalent_diameter(area))
    if diameters:
        result["pupil_px"] = round(float(np.mean(diameters)), 2)
    return result

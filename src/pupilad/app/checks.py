import cv2
import numpy as np

from pupilad.webcam.pupil_detector import (
    detect_eyes,
    detect_faces,
    equivalent_diameter,
    eye_contrast,
    pupil_area,
)

MIN_FACE_FRACTION = 0.12


def brightness(gray):
    return round(float(gray.mean()), 1) if gray.size else None


def inspect_frame(jpeg_bytes):
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the image")
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_side = max(40, int(min(width, height) * MIN_FACE_FRACTION))
    faces = detect_faces(gray, min_size=(min_side, min_side))
    result = {
        "width": width,
        "height": height,
        "faces": len(faces),
        "face": None,
        "eyes": 0,
        "eye_y": None,
        "eye_width_px": None,
        "eye_contrast": None,
        "pupil_px": None,
        "frame_brightness": brightness(gray),
        "face_brightness": None,
        "background_brightness": None,
        "face_left_brightness": None,
        "face_right_brightness": None,
    }
    if not faces:
        return result
    face_x, face_y, face_w, face_h = faces[0]
    result["face"] = {
        "x": face_x / width,
        "y": face_y / height,
        "w": face_w / width,
        "h": face_h / height,
    }
    face = gray[face_y : face_y + face_h, face_x : face_x + face_w]
    background = np.ones(gray.shape, dtype=bool)
    background[face_y : face_y + face_h, face_x : face_x + face_w] = False
    result["face_brightness"] = brightness(face)
    result["background_brightness"] = brightness(gray[background])
    result["face_left_brightness"] = brightness(face[:, : face_w // 2])
    result["face_right_brightness"] = brightness(face[:, face_w // 2 :])
    eyes = detect_eyes(face)
    result["eyes"] = len(eyes)
    if eyes:
        centers = [face_y + eye_y + eye_h / 2 for _, eye_y, _, eye_h in eyes]
        result["eye_y"] = round(float(np.mean(centers)) / height, 3)
        result["eye_width_px"] = int(np.mean([eye_w for _, _, eye_w, _ in eyes]))
    else:
        result["eye_y"] = round((face_y + face_h * 0.4) / height, 3)
    diameters = []
    contrasts = []
    for eye_x, eye_y, eye_w, eye_h in eyes:
        region = face[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w]
        contrasts.append(eye_contrast(region))
        area = pupil_area(region)
        if area > 0:
            diameters.append(equivalent_diameter(area))
    if contrasts:
        result["eye_contrast"] = round(float(np.mean(contrasts)), 1)
    if diameters:
        result["pupil_px"] = round(float(np.mean(diameters)), 2)
    return result

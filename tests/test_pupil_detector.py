import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pupilad.webcam.pupil_detector import equivalent_diameter, pupil_area


def synthetic_eye(pupil_radius, pupil_level, iris_level=90, skin_level=150, size=64):
    eye = np.full((size, size), skin_level, dtype=np.uint8)
    center = (size // 2, int(size * 0.55))
    cv2.circle(eye, center, int(size * 0.2), iris_level, -1)
    cv2.circle(eye, center, pupil_radius, pupil_level, -1)
    return eye


def test_pupil_found_when_it_is_darker_than_iris_but_not_black():
    area = pupil_area(synthetic_eye(pupil_radius=5, pupil_level=55))
    assert area > 0
    assert equivalent_diameter(area) == pytest.approx(10, abs=3)


def test_no_pupil_in_flat_low_contrast_eye():
    eye = np.full((64, 64), 120, dtype=np.uint8)
    cv2.circle(eye, (32, 35), 10, 114, -1)
    assert pupil_area(eye) == 0.0

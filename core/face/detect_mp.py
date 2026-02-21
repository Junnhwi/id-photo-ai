import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join("models", "blaze_face_short_range.tflite")

_detector = None


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.6
    )

    _detector = mp_vision.FaceDetector.create_from_options(options)
    return _detector


def detect_faces_mediapipe(image_bgr):
    """
    반환:
      [
        {"bbox": (x, y, w, h), "score": float},
        ...
      ]
    """
    detector = _get_detector()

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ✅ 여기 핵심: MpImage가 아니라 mp.Image를 써야 함
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    faces = []
    if result and result.detections:
        for det in result.detections:
            score = float(det.categories[0].score) if det.categories else 0.0
            b = det.bounding_box  # origin_x, origin_y, width, height (픽셀)

            faces.append({
                "bbox": (int(b.origin_x), int(b.origin_y), int(b.width), int(b.height)),
                "score": score
            })

    return faces
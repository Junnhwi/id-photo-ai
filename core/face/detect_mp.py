import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

def detect_faces_mediapipe(image_bgr):
    """
    FaceMesh 기반 얼굴 탐지.
    반환:
      [{"bbox": (x, y, w, h), "score": 1.0}, ...]
    """

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = mp_face_mesh.process(rgb)

    faces = []

    if not results.multi_face_landmarks:
        return faces

    for face_landmarks in results.multi_face_landmarks:
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]

        x_min = int(min(xs) * w)
        x_max = int(max(xs) * w)
        y_min = int(min(ys) * h)
        y_max = int(max(ys) * h)

        faces.append({
            "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            "score": 1.0  # FaceMesh는 별도 score 없음
        })

    return faces
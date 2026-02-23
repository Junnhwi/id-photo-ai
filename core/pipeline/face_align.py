import os
import cv2
import numpy as np
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def _get_eye_centers(landmarks, image_w, image_h):
    """
    MediaPipe FaceMesh 랜드마크에서
    왼쪽/오른쪽 눈 중심 좌표를 계산한다.
    """

    # MediaPipe 눈 랜드마크 인덱스
    LEFT_EYE_IDX = [33, 133]
    RIGHT_EYE_IDX = [362, 263]

    left_eye = np.mean([
        (landmarks[i].x * image_w, landmarks[i].y * image_h)
        for i in LEFT_EYE_IDX
    ], axis=0)

    right_eye = np.mean([
        (landmarks[i].x * image_w, landmarks[i].y * image_h)
        for i in RIGHT_EYE_IDX
    ], axis=0)

    return left_eye, right_eye


def align_and_crop_face(image_bgr, output_size=512):
    """
    1) 얼굴 랜드마크 추출
    2) 눈 기준으로 회전 보정
    3) 정면 기준으로 크롭
    4) 정사각형으로 리사이즈
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None  # 정렬 실패

    h, w = image_bgr.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark

    # -------------------------
    # 1️⃣ 눈 중심 계산
    # -------------------------
    left_eye, right_eye = _get_eye_centers(landmarks, w, h)

    # -------------------------
    # 2️⃣ 회전 각도 계산
    # -------------------------
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # -------------------------
    # 3️⃣ 회전 행렬 생성
    # -------------------------

    ex, ey = np.mean([left_eye, right_eye], axis=0)
    eyes_center = (int(ex), int(ey))
    M = cv2.getRotationMatrix2D(eyes_center, float(angle), 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

    # -------------------------
    # 4️⃣ 얼굴 중심 기준 크롭
    # -------------------------
    # 코 중심 (landmark 1번 사용)
    nose = landmarks[1]
    nose_x = int(nose.x * w)
    nose_y = int(nose.y * h)

    crop_size = int(w * 0.6)

    x1 = max(nose_x - crop_size // 2, 0)
    y1 = max(nose_y - crop_size // 2, 0)
    x2 = min(x1 + crop_size, w)
    y2 = min(y1 + crop_size, h)

    face_crop = rotated[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    # -------------------------
    # 5️⃣ 고정 해상도 리사이즈
    # -------------------------
    face_crop = cv2.resize(face_crop, (output_size, output_size))

    return face_crop
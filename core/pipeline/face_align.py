import cv2
import numpy as np
import mediapipe as mp


# FaceMesh는 "얼굴 랜드마크(468개)"를 뽑아주는 모델
# static_image_mode=True : 사진(정지 이미지) 처리에 적합
# max_num_faces=1 : 우리는 증명사진 후보니까 1명만 대상으로 함
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def _get_eye_centers(landmarks, image_w, image_h):
    """
    FaceMesh 랜드마크에서 왼쪽/오른쪽 눈의 중심점을 계산한다.
    landmarks[i].x, landmarks[i].y 는 0~1 사이의 "정규화 좌표"이므로
    실제 픽셀 좌표로 바꾸려면 이미지 크기(w,h)를 곱해야 한다.
    """
    LEFT_EYE_IDX = [33, 133]
    RIGHT_EYE_IDX = [362, 263]

    left_eye = np.mean(
        [(landmarks[i].x * image_w, landmarks[i].y * image_h) for i in LEFT_EYE_IDX],
        axis=0,
    )
    right_eye = np.mean(
        [(landmarks[i].x * image_w, landmarks[i].y * image_h) for i in RIGHT_EYE_IDX],
        axis=0,
    )

    return left_eye, right_eye


def draw_landmarks(image_bgr, landmarks, radius=1):
    """
    얼굴 랜드마크를 이미지 위에 점으로 그려서 확인용 이미지를 만든다.
    (디버깅/포트폴리오용. 학습에는 필수 아님)
    """
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(out, (x, y), radius, (0, 255, 0), -1)

    return out


def align_and_crop_face(image_bgr, output_size=512, crop_ratio=0.6):
    """
    반환:
      (face_crop, landmarks) 또는 (None, None)

    동작:
    1) FaceMesh로 랜드마크(468개) 추출
    2) 두 눈의 기울기를 계산해서 회전(수평 맞춤)
    3) 코(landmark 1) 중심 기준으로 정사각형 크롭
    4) output_size로 리사이즈
    """

    if image_bgr is None:
        return None, None

    h, w = image_bgr.shape[:2]

    # 1) 랜드마크 추출
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark

    # 2) 눈 중심 → 각도 계산
    left_eye, right_eye = _get_eye_centers(landmarks, w, h)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 3) 회전 (OpenCV는 center가 "파이썬 숫자 튜플"이어야 안전)
    ex, ey = np.mean([left_eye, right_eye], axis=0)
    eyes_center = (int(ex), int(ey))

    M = cv2.getRotationMatrix2D(eyes_center, float(angle), 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

    # ⚠ 지금은 rotated 이미지 기준으로 다시 랜드마크를 뽑지 않았다.
    # (MVP 단계에서는 충분. 다음 단계에서 더 정교하게 개선 가능)

    # 4) 코 중심 기준 크롭 (정사각형)
    nose = landmarks[1]
    nose_x = int(nose.x * w)
    nose_y = int(nose.y * h)

    crop_size = int(w * float(crop_ratio))

    x1 = max(nose_x - crop_size // 2, 0)
    y1 = max(nose_y - crop_size // 2, 0)
    x2 = min(x1 + crop_size, w)
    y2 = min(y1 + crop_size, h)

    face_crop = rotated[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    # 5) 고정 크기 리사이즈
    face_crop = cv2.resize(face_crop, (output_size, output_size))

    return face_crop, landmarks
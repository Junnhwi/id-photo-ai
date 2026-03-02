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
    FaceMesh 랜드마크에서 왼쪽/오른쪽 눈의 중심점을 계산한다.
    landmarks[i].x, landmarks[i].y 는 0~1 정규화 좌표이므로 픽셀로 변환한다.
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
    """디버깅용: 원본 이미지 위에 랜드마크 점을 찍어 저장할 때 사용"""
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(out, (x, y), radius, (0, 255, 0), -1)
    return out


def frame_id_photo(
    image_bgr,
    *,
    out_w=600,
    out_h=800,
    eye_y_ratio=0.35,
    eye_dist_to_crop_w=2.3,
):
    """
    Day-5: 증명사진 비율 고정 프레이밍(3:4)

    반환:
      (framed_bgr, landmarks) 또는 (None, None)

    핵심 아이디어(중요):
    - 눈 사이 거리(eye_distance)를 '얼굴 스케일 기준'으로 사용한다.
    - 최종 사진에서 '눈 위치'를 항상 out_h의 eye_y_ratio 지점으로 고정한다.
    - crop 영역은 3:4 비율(가로:세로=out_w:out_h)을 유지한다.

    파라미터 설명:
    - out_w/out_h: 최종 증명사진 픽셀 크기 (기본 600x800)
    - eye_y_ratio: 최종 이미지에서 눈이 위치할 세로 비율 (0.35 = 위에서 35%)
    - eye_dist_to_crop_w: crop 가로폭을 '눈 사이 거리'의 몇 배로 할지 (크면 얼굴 작아짐)
      * 대략 2.1~2.6 사이에서 튜닝
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

    # 2) 눈 좌표(픽셀) 계산 + 회전 각도 계산
    left_eye, right_eye = _get_eye_centers(landmarks, w, h)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 3) 회전(눈 수평 맞추기)
    ex, ey = np.mean([left_eye, right_eye], axis=0)
    eyes_center = (int(ex), int(ey))
    M = cv2.getRotationMatrix2D(eyes_center, float(angle), 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

    # (중요) 여기서는 'rotated' 기준 랜드마크를 다시 계산하지 않는다(MVP).
    # 프레이밍은 눈 위치(eyes_center)와 눈 거리(eye_distance)를 기반으로 한다.

    # 4) 프레이밍 crop 크기 결정
    eye_distance = float(np.linalg.norm(np.array(right_eye) - np.array(left_eye)))
    if eye_distance <= 1.0:
        return None, None

    crop_w = int(eye_distance * float(eye_dist_to_crop_w))
    crop_h = int(crop_w * (out_h / out_w))  # 3:4 유지

    print(
        f"[framing] param={eye_dist_to_crop_w} eye_dist={eye_distance:.1f} "
        f"crop_w={crop_w} crop_h={crop_h}"
    )

    # 5) crop 중심 좌표 계산
    # 눈이 최종 이미지에서 out_h * eye_y_ratio 위치에 오도록 한다.
    # 즉, crop의 top = eyes_center_y - crop_h * eye_y_ratio
    cx = int(eyes_center[0])
    cy = int(eyes_center[1])

    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h * float(eye_y_ratio))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    print(
        f"[framing] crop box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
        f"img_w={w}, img_h={h}"
    )

    # 6) 이미지 밖으로 나가는 부분을 보정(클램프)
    #   - out of bounds가 있으면 crop 영역을 안쪽으로 밀어 넣는다.
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
        if x1 < 0:
            x1 = 0
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h
        if y1 < 0:
            y1 = 0

    # 7) crop (패딩 포함: 항상 crop_h x crop_w 보장)
    # 원하는 crop 영역: [x1:x2, y1:y2] (크기 crop_w x crop_h)
    # rotated에서 실제로 겹치는 영역만 잘라서, 빈 캔버스에 붙인다.
    canvas = np.zeros((crop_h, crop_w, 3), dtype=rotated.dtype)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    # canvas에 붙일 위치(원래 좌표계 기준으로 이동)
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # 겹치는 영역이 아예 없으면 실패
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return None, None

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = rotated[src_y1:src_y2, src_x1:src_x2]
    face_crop = canvas  

    # 8) 최종 리사이즈 (600x800)
    framed = cv2.resize(face_crop, (out_w, out_h))

    return framed, landmarks
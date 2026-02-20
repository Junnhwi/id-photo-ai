# 얼굴 탐지 함수

import cv2


def detect_faces_opencv(image_bgr):
    """
    OpenCV Haar Cascade(전통적인 얼굴 탐지 모델)를 이용해 얼굴을 탐지한다.

    입력:
      - image_bgr: OpenCV로 읽은 이미지 (BGR 채널 순서)

    출력:
      - faces: 얼굴 박스 리스트
        각 박스는 (x, y, w, h)
        x, y: 얼굴 사각형의 왼쪽 위 좌표
        w, h: 사각형의 너비/높이
    """

    # 1) OpenCV에 내장된 얼굴 탐지 모델(haar cascade) 파일 경로
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # 2) 모델 로드 (얼굴 탐지기 생성)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 3) 얼굴 탐지는 보통 흑백 이미지에서 더 잘 동작하고 더 빠르다.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 4) 얼굴 탐지 실행
    # - scaleFactor: 이미지를 줄여가며 탐지할 때 줄이는 비율(보통 1.1~1.3)
    # - minNeighbors: 얼굴 후보가 여러 번 겹쳐 검출될 때 "얼굴"로 확정하는 엄격도 (높을수록 오탐 감소)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    return faces
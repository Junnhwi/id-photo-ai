# 파일 저장과 폴더 구조(Job 시스템)를 담당하는 핵심 모듈
# app/api_routes.py에서 저장이 필요할 때마다 여기 함수들을 호출
import os
from datetime import datetime
import uuid
import shutil

# Job 폴더들이 저장될 기본 경로
BASE_JOBS_DIR = "data/jobs"


def create_job_folder() -> tuple[str, str]:
    """
    job 폴더를 새로 만들고, (job_id, job_path)를 반환한다.

    - job_id: 폴더 이름으로 사용할 고유 문자열
    - job_path: 실제로 생성된 폴더 경로 (예: data/jobs/<job_id>)
    """

    # 1) 현재 시간을 문자열로 만들기
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 예: 20260217_193012

    # 2) 랜덤 고유 값 생성.
    random_part = str(uuid.uuid4())[:8]

    # 3) 최종 job_id(폴더 이름)생성
    # 예: 20260217_193012_a1b2c3d4
    job_id = f"{timestamp}_{random_part}"

    # 4) job 폴더 경로 만들기
    job_path = os.path.join(BASE_JOBS_DIR, job_id)

    # 5) 폴더 생성 (uploads, outputs 포함)
    uploads_dir = os.path.join(job_path, "uploads")
    outputs_dir = os.path.join(job_path, "outputs")

    os.makedirs(uploads_dir, exist_ok=True) # 폴더가 이미 있어도 에러 내지 말고 그냥 넘어가라는 뜻
    os.makedirs(outputs_dir, exist_ok=True)

    #얼굴 탐지 결과를 나중에 “디버깅 이미지(얼굴 박스 표시)”로 저장할 수 있도록 work 폴더도 만듦
    uploads_dir = os.path.join(job_path, "uploads")
    outputs_dir = os.path.join(job_path, "outputs")
    work_dir = os.path.join(job_path, "work")

    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    

    return job_id, job_path

import pathlib

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"} # 허용되는 이미지 확장자 목록


def is_allowed_image(filename: str) -> bool:
    """
    파일 확장자가 이미지 허용 목록에 있는지 검사한다.
    """
    ext = pathlib.Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def safe_filename(original_name: str) -> str:
    """
    파일명이 겹칠 수 있으므로, UUID를 붙여서 안전한 파일명으로 만든다.

    예:
    - 원본: selfie.png
    - 저장: selfie__a1b2c3d4.png
    """
    ext = pathlib.Path(original_name).suffix.lower()
    stem = pathlib.Path(original_name).stem

    unique = str(uuid.uuid4())[:8]
    return f"{stem}__{unique}{ext}"


def save_upload_file(upload_file, save_dir: str) -> str:
    """
    UploadFile을 디스크에 저장하고, 저장된 파일명을 반환한다.
    """
    os.makedirs(save_dir, exist_ok=True)

    filename = safe_filename(upload_file.filename)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return filename

import os
import shutil
from typing import List

from fastapi import APIRouter, UploadFile, File

from core.io.storage import create_job_folder

router = APIRouter()


@router.post("/api/jobs")
async def create_job(files: List[UploadFile] = File(...)):
    """
    여러 이미지 파일을 업로드 받아서 Job 폴더에 저장한다.

    요청:
    - files: 이미지 파일 여러 개

    응답:
    - job_id: 생성된 작업 ID
    - saved_files: 저장된 파일 이름 목록
    """

    # 1) job 폴더 생성
    job_id, job_path = create_job_folder()

    # 2) 업로드 파일을 저장할 경로
    uploads_dir = os.path.join(job_path, "uploads")

    saved_files = []

    # 3) 업로드된 파일을 하나씩 저장
    for file in files:
        save_path = os.path.join(uploads_dir, file.filename)

        # "wb" = write binary (바이너리로 파일 저장)
        with open(save_path, "wb") as buffer:
            # file.file(업로드 데이터)을 buffer(새 파일)로 복사
            shutil.copyfileobj(file.file, buffer)

        saved_files.append(file.filename)

    return {
        "job_id": job_id,
        "saved_files": saved_files,
    }

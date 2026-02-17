import os
import json
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from core.io.storage import (
    create_job_folder,
    is_allowed_image,
    save_upload_file,
)

router = APIRouter()


@router.post("/api/jobs")
async def create_job(files: List[UploadFile] = File(...)):
    """
    여러 이미지 파일을 업로드 받아서 Job 폴더에 저장한다.
    - 이미지 확장자 검사
    - 저장 파일명 충돌 방지
    - report.json 뼈대 생성
    """

    # 1) 파일이 하나도 안 들어오면 에러 처리
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # 2) job 폴더 생성
    job_id, job_path = create_job_folder()
    uploads_dir = os.path.join(job_path, "uploads")

    saved_files = []
    rejected_files = []

    # 3) 파일 하나씩 검사 및 저장
    for f in files:
        if not is_allowed_image(f.filename):
            rejected_files.append(
                {"filename": f.filename, "reason": "Unsupported file extension"}
            )
            continue

        saved_name = save_upload_file(f, uploads_dir)
        saved_files.append(saved_name)

    # 4) 저장된 이미지가 0개면 job 자체를 실패로 처리
    if len(saved_files) == 0:
        raise HTTPException(
            status_code=400,
            detail="All uploaded files were rejected. Please upload jpg/png/webp images.",
        )

    # 5) report.json 뼈대 생성
    report = {
        "job_id": job_id,
        "summary": {
            "total_received": len(files),
            "saved": len(saved_files),
            "rejected": len(rejected_files),
        },
        "saved_files": saved_files,
        "rejected_files": rejected_files,
        "next_stage": "quality_check (planned)",
    }

    report_path = os.path.join(job_path, "report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    return {
        "job_id": job_id,
        "saved_files": saved_files,
        "rejected_files": rejected_files,
    }

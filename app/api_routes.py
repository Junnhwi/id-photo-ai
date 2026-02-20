# 실제 API 기능이 들어있는 파일
import os
import json
import cv2

from core.face.detect import detect_faces_opencv
from core.report.update_report import load_report, save_report

from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from core.io.storage import (
    create_job_folder,
    is_allowed_image,
    save_upload_file,
)

router = APIRouter()


@router.post("/api/jobs") # 업로드 요청을 받는 API 엔드포인트. 여러 이미지 파일을 업로드 받아서 Job 폴더에 저장한다.
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
    # ----------------------------
    # 1) 업로드 결과 report 생성 (Day-2)
    # ----------------------------
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

    # ----------------------------
    # 2) 얼굴 탐지 품질 검사 (Day-3) 
    # ----------------------------
    report = load_report(report_path)

    passed_images = []
    rejected_images = []

    for filename in saved_files:
        img_path = os.path.join(uploads_dir, filename)

        image = cv2.imread(img_path)    #저장된 이미지 파일을 읽어서, 메모리 안에 “이미지 배열”로 만든다 

        if image is None:               #이미지를 읽는 데 실패한 경우 None을 반환 
            rejected_images.append(
                {"filename": filename, "reason": "Failed to read image"}
            )
            continue


        #OpenCV Haar Cascade(얼굴 탐지 모델)를 이용해 얼굴을 탐지한다.
        faces = detect_faces_opencv(image)    
        face_count = int(len(faces))
        
        # 얼굴이 0개면 거부
        if face_count == 0:
            rejected_images.append(
                {"filename": filename, "reason": "No face detected"}
            )

        # 얼굴이 2개 이상이면 거부 (새 규칙)
        elif face_count >= 2:
            rejected_images.append(
                {"filename": filename, "reason": "Multiple faces detected", "faces_detected": face_count}
            )

        # 얼굴이 정확히 1개면 통과
        else:
            passed_images.append(
                {"filename": filename, "faces_detected": face_count}
            )

            
    report["quality_check"] = {
        "passed": passed_images,
        "rejected": rejected_images
    }

    report["summary"]["quality_passed"] = len(passed_images)
    report["summary"]["quality_rejected"] = len(rejected_images)

    report["next_stage"] = "blur/brightness/face_size checks (planned)"

    save_report(report_path, report)


    # ----------------------------
    # 3) 최종 응답
    # ----------------------------
    return {
        "job_id": job_id,
        "saved_files": saved_files,
        "rejected_files": rejected_files,
        "quality_check": report["quality_check"],
    }


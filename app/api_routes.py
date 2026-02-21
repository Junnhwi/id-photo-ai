# 실제 API 기능이 들어있는 파일
import os
import json
import cv2

from core.report.update_report import load_report, save_report
from core.face.detect_mp import detect_faces_mediapipe
from core.face.visualize import save_face_preview

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
    work_dir = os.path.join(job_path, "work")

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

        preview_name = f"preview__{filename}"

        if image is None:
            rejected_images.append({
                "filename": filename,
                "reason": "Failed to read image",
                "preview": None
            })
            continue


        #(얼굴 탐지 모델)를 이용해 얼굴을 탐지한다.
        faces_raw = detect_faces_mediapipe(image)

        # 1차: 기본 threshold (정상 사진에서 오탐 억제)
        faces = [f for f in faces_raw if f.get("score", 0.0) >= 0.6]

        # 2차(fallback): 0개면 너무 보수적일 수 있으니 threshold를 낮춰 다시 시도
        if len(faces) == 0 and len(faces_raw) > 0:
            faces = [f for f in faces_raw if f.get("score", 0.0) >= 0.3]

        face_count = len(faces)

        # 대표 얼굴 1개 선택: 일단 가장 큰 박스를 대표로(간단 버전)
        main_face_bbox = None
        if face_count > 0:
            main_face = max(
                faces,
                key=lambda f: (f["bbox"][2] * f["bbox"][3]) * f.get("score", 1.0)
            )
            main_face_bbox = main_face["bbox"]

        # preview 저장
        save_face_preview(
            image,
            faces,
            main_face_bbox,
            save_dir=work_dir,
            filename=filename,
        )
        
        # 얼굴이 0개면 거부
        if face_count == 0:
            rejected_images.append({
                "filename": filename,
                "reason": "No face detected",
                "faces_detected": face_count,
                "preview": preview_name
            })

        # 얼굴이 2개 이상이면 거부 (새 규칙)
        elif face_count >= 2:
            rejected_images.append({
                "filename": filename,
                "reason": "Multiple faces detected",
                "faces_detected": face_count,
                "preview": preview_name
            })

        # 얼굴이 정확히 1개면 통과
        else:
            passed_images.append({
                "filename": filename,
                "faces_detected": face_count,
                "preview": preview_name
            })


            
    report["quality_check"] = {
        "passed": passed_images,
        "rejected": rejected_images
    }

    report["summary"]["quality_passed"] = len(passed_images)
    report["summary"]["quality_rejected"] = len(rejected_images)

    # ----------------------------
    # (Day-3 Gate) 최소 통과 장수 기준
    # ----------------------------
    MIN_PASSED = 10  # 나중에 config로 빼도 됨

    passed_count = report["summary"]["quality_passed"]
    can_proceed = passed_count >= MIN_PASSED

    report["policy"] = {
        "min_passed_required": MIN_PASSED,
        "recommended_upload_range": "10~30"
    }

    report["gate"] = {
        "can_proceed": can_proceed,
        "passed_count": passed_count,
        "missing_count": max(0, MIN_PASSED - passed_count),
        "message": (
            "OK to proceed to next stage."
            if can_proceed
            else f"Not enough valid photos. Please upload at least {max(0, MIN_PASSED - passed_count)} more."
        )
    }

    # next_stage도 gate 결과에 따라 다르게
    report["next_stage"] = "background/crop/resize (planned)" if can_proceed else "upload_more_photos"
    
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
        "policy": report["policy"],
        "gate": report["gate"],
    }


import os
from datetime import datetime
import uuid

# Job 폴더들이 저장될 "기본 경로"
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

    return job_id, job_path

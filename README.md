# ID Photo AI

AI 기반 증명사진 자동 생성 프로젝트입니다.

---

## 프로젝트 소개

사용자가 사진 여러 장을 업로드하면, 파이프라인이 자동으로 얼굴 검출/정렬/배경 합성/임베딩/데이터셋 생성을 수행합니다.
현재 파이프라인은 **retouch 단계 없이** 동작합니다.

---

## 현재 파이프라인 순서

1. `POST /api/jobs` : 업로드 + 품질 검사
2. `POST /api/jobs/{job_id}/prepare_faces` : 증명사진 프레이밍
3. `POST /api/jobs/{job_id}/background` : 배경 제거 + 흰 배경 합성
4. `POST /api/jobs/{job_id}/embedding` : 동일인 임베딩 필터링
5. `POST /api/jobs/{job_id}/build_dataset` : 학습 데이터셋 생성

> `retouch` 엔드포인트/모듈은 제거되었습니다.

---

## 실행 방법

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## 충돌/에러 예방 Git 워크플로우

원격과 로컬이 갈라졌을 때(`diverged`)는 아래 순서를 권장합니다.

```bash
git fetch origin
git pull --rebase origin main
```

충돌이 나면:

```bash
# 충돌 마커 탐색
rg -n '<<<<<<<|=======|>>>>>>>' app core

# 파일 수정 후
git add <resolved_files>
git rebase --continue
```

작업 전 기본 점검:

```bash
git status
python -m compileall app core
```

---

## 기술 스택

- Python
- FastAPI
- OpenCV
- MediaPipe
- ONNX Runtime

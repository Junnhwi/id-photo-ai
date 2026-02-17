<<<<<<< HEAD
# ID Photo AI

AI 기반 증명사진 자동 생성 프로젝트

---

## 프로젝트 소개

일상 사진 여러 장을 업로드하면,  
AI 기반 자동 처리 과정을 통해 증명사진 규격에 맞는 결과물을 생성하는 프로젝트.

사용자는 스튜디오 촬영 없이도  
정면, 단색 배경, 규격 크기 조건을 만족하는 증명사진을 얻을 수 있음.

---

## 프로젝트 목표

1. 사진 10~30장 업로드
2. 품질 검사 (흐림, 정면 여부, 얼굴 크기 등)
3. 얼굴 정렬 (기울기 보정)
4. 배경 단색 합성
5. 규격 크롭 및 리사이즈
6. 최종 결과 및 처리 리포트 생성

---

## 개발 목적

- 컴퓨터 비전(Computer Vision) 파이프라인 설계 경험
- AI 기반 이미지 처리 시스템 구조 이해
- FastAPI 기반 서버 개발
- 실제 서비스 구조에 가까운 Job 시스템 설계

---

## 기술 스택

- Python
- FastAPI
- OpenCV
- Pillow
- NVIDIA RTX 3060 Ti (개발자 GPU 환경)

---

## 실행 방법

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload ## 서버 시작


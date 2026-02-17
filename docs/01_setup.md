# 01 - Environment Setup

이 문서는 ID Photo AI 프로젝트를 실행하기 위한 환경 설정 과정을 설명한다.

---

## 1. 프로젝트 폴더 생성

1. "id-photo-ai" 폴더 생성
2. VS Code 실행
3. File → Open Folder → id-photo-ai 선택

---

## 2. 가상환경(Virtual Environment) 생성

가상환경은 프로젝트 전용 Python 실행 공간이다.

다른 프로젝트와 라이브러리 충돌을 방지하기 위해 사용한다.

### 가상환경 생성

```bash
python -m venv venv

venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload

http://127.0.0.1:8000/          #브라우저 접속
http://127.0.0.1:8000/docs      #API 문서 확인
from fastapi import FastAPI

from app.api_routes import router

app = FastAPI() # 서버 앱 생성

app.include_router(router)  # api_routes.py에 있는 router(API 모음)를 서버에 등록

@app.get("/")   # 루트 경로에 GET 요청이 들어오면 서버 상태 메시지를 반환(서버가 살아있는지 확인하는 기본 엔드포인트)
def root():
    return {"message": "ID Photo AI Server Running"}

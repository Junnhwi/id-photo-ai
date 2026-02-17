from fastapi import FastAPI

from app.api_routes import router

app = FastAPI()

# api_routes.py에 있는 router를 서버에 등록
app.include_router(router)


@app.get("/")
def root():
    return {"message": "ID Photo AI Server Running"}

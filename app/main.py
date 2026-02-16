from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "ID Photo AI Server Running"}

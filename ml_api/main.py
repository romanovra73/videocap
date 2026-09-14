from fastapi import FastAPI
from routers import yolo_api
from routers import pool_detect
from routers import ch1_tracker

app = FastAPI(
    title="Machine learning API",
    description="API для обработки данных методами машинного обучения",
    version="1.0"
)

app.include_router(yolo_api.router)
app.include_router(pool_detect.router)
app.include_router(ch1_tracker.router)

@app.get("/")
def root():
    return {"message": "Welcome to ML API. Folow /docs for documentation."}
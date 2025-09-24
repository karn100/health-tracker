from fastapi import FastAPI
from app.routes import health

app = FastAPI(
    title="Health Tracker API",
    version = "1.0",
    description="API for Personalised Health Tracking"
)

app.include_router(health.router, prefix="/health", tags=["Health"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Health Tracker API.."}
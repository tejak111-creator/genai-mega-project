#API/Presentation Layer part of Health check route
from fastapi import APIRouter
from app.core.models import HealthResponse
from app.core.config import settings 

router = APIRouter(tags=["health"])
#tags part groups this endpoint under "health" in Swagger UI
#A mini fastAPI app, that you later plug into the main app
'''
APIRouter lets you organize routes into modules 
instead of putting everything in main.py
'''
#Define the route, register a get endpoint
@router.get("/health", response_model=HealthResponse)
def health()-> HealthResponse:
    return HealthResponse(service=settings.app_name)
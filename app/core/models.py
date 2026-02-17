#DATA VALIDATION LAYER
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
#THIS defines the structure of a healthcheck response
#this means status must be a str and its default value is ok
#service required string field
#is service is missing,validation error
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1,max_length=8000)
    #... means mandatory
    #This defines what client must send
class ChatResponse(BaseModel):
    response: str
    request_id: str
#THIS defines what your API will return


#FastAPI will: vaildte response shape,serialize to JSON,Document in Swagger UI

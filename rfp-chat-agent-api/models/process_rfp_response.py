from pydantic import BaseModel

class ProcessRfpResponse(BaseModel):
    Pursuit_Name: str
    Decision_Log: dict
from pydantic import BaseModel
from typing import Optional
from models.decision_log import DecisionLog

class ProcessRfpResponse(BaseModel):
    Pursuit_Name: str
    Decision_Log: Optional[DecisionLog] = None
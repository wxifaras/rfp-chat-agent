from pydantic import BaseModel
from models.decision_log import DecisionLog

class ProcessRfpResponse(BaseModel):
    Pursuit_Name: str
    Decision_Log: DecisionLog
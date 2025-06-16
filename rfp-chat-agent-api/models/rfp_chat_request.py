from pydantic import BaseModel

class RfpChatRequest(BaseModel):
    User_Id: str
    Session_Id: str
    Message: str
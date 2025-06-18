from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional
import uuid

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: str = Field(..., description="Chat session identifier")
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    message: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp (UTC, timezone-aware)"
    )
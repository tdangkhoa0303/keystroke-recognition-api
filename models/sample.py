from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone


class Event(BaseModel):
    key: str
    direction: int  # 0 for DOWN, 1 for UP
    timestamp: int


class Sample(BaseModel):
    _id: Optional[str]
    userId: str
    sessionId: Optional[str]
    events: List[Event]
    createdAt: Optional[datetime] = datetime.now(timezone.utc)

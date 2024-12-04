from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional


class History(BaseModel):
    _id: str
    userId: int
    accuracy: Optional[int]
    numberOfSamples: int
    createdAt: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))
    trainedAt: Optional[datetime]

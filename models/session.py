from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional


class Session(BaseModel):
    _id: str
    userId: str
    createdAt: Optional[datetime] = datetime.now(timezone.utc)
    updatedAt: Optional[datetime] = datetime.now(timezone.utc)
    expireAt: datetime
    ua: str

from datetime import datetime, timezone

from pydantic import BaseModel, EmailStr


class User(BaseModel):
    _id: str
    lastName: str
    firstName: str
    email: EmailStr
    createdAt: datetime = datetime.now(timezone.utc)

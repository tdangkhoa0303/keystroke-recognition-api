from pydantic import BaseModel
from typing import Optional, Any


class Response(BaseModel):
    success: bool
    message: str
    statusCode: int
    data: Optional[Any] = None

from datetime import datetime
from typing import List
import math

from fastapi import APIRouter, Depends, status, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


from dependencies import auth
from helpers.trainer import predict_user_samples, train_model_for_user
from vendors.supabase import supabase

router = APIRouter()


class BaseSample(BaseModel):
    events: List[dict]
    createdAt: datetime


class StoreSamplesPayload(BaseModel):
    samples: List[BaseSample]


def query_params(
    user_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    is_revoked: Optional[bool] = Query(None),
    page: int = Query(1, gt=0),
    page_size: int = Query(10, ge=1, le=100),
):
    return {
        "user_id": user_id,
        "start_date": start_date,
        "end_date": end_date,
        "is_revoked": is_revoked,
        "page": page,
        "page_size": page_size,
    }


# FastAPI route
@router.get("/")
def get_sessions(params: dict = Depends(query_params)):
    try:
        user_id = params["user_id"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        is_revoked = params["is_revoked"]
        page = params["page"]
        page_size = params["page_size"]

        query = (
            supabase.from_("session_metadata")
            .select(
                "id, ua, ip, created_at, is_revoked",
                "user:profiles(*)",
                "samples(total_legitimate:is_legitimate::int.sum(), total_samples:id.count())",
                count="exact",
            )
            .range((page - 1) * page_size, page * page_size - 1)
            .order("created_at", desc=True)
        )

        # Apply filters to count query
        if user_id:
            query = query.eq("user_id", user_id)
        if start_date:
            query = query.gte("created_at", start_date.isoformat())
        if end_date:
            query = query.lte("created_at", end_date.isoformat())
        if is_revoked:
            query = query.eq("is_revoked", is_revoked)

        response = query.execute()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "items": response.data,
                "total_items": response.count,
                "page": page,
                "page_size": page_size,
                "total_pages": math.ceil(response.count / page_size),
            },
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/revoke")
async def revoke_session(session_id):
    supabase.table("session_metadata").update({"is_revoked": True}).eq(
        "id", session_id
    ).execute()

    return JSONResponse(
        content={},
        status_code=status.HTTP_200_OK,
    )

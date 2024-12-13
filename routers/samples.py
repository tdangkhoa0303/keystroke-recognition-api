from datetime import datetime
from typing import List, Optional
import math

from fastapi import APIRouter, Depends, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dependencies import auth
from vendors.supabase import supabase
from helpers.trainer import predict_user_samples, train_model_for_user

router = APIRouter()


class BaseSample(BaseModel):
    events: List[dict]
    createdAt: datetime


class StoreSamplesPayload(BaseModel):
    samples: List[BaseSample]


@router.post("/train")
async def train(user=Depends(auth.extract)):
    await train_model_for_user(user)

    return JSONResponse(
        content={"message": "Your train request is queued"},
        status_code=status.HTTP_201_CREATED,
    )


class VeriySamplesPayload(BaseModel):
    samples: List[BaseSample]


@router.post("/verify")
async def verify(
    payload: VeriySamplesPayload,
    user=Depends(auth.extract),
    profile=Depends(auth.extract_profile),
):
    verifying_samples = payload.samples
    result = await predict_user_samples(
        user, profile=profile, samples=map(lambda x: x.model_dump(), verifying_samples)
    )

    return JSONResponse(
        content={"is_legitimate": result},
        status_code=status.HTTP_201_CREATED,
    )


def query_params(
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    page: Optional[int] = Query(None),
    page_size: Optional[int] = Query(None)   
):
    return {
        "user_id": user_id,
        "session_id": session_id,
        "page": page,
        "page_size": page_size,
    }


@router.get("/")
async def get_samples(params: dict = Depends(query_params)):
    query = supabase.from_("samples").select(
        "id, created_at, predicted_score, security_level, created_at, events",
        "session_metadata(ua, ip, created_at)",
        "user:profiles(id, email, first_name, last_name)",
        count="exact",
    ).order("created_at", desc=True)

    if params["user_id"]:
        query = query.eq("user_id", params["user_id"])

    if params["session_id"]:
        query = query.eq("session_id", params["session_id"])
    
    if params["page"] and params["page_size"]:
        page = params["page"] or 1
        page_size = params["page_size"] or 10
        query = query.range((page - 1) * page_size, page * page_size - 1)

    response = query.execute()

    return JSONResponse(
        content={
            "items": response.data,
            "total_items": response.count,
            "page": params["page"],
            "page_size": params["page_size"],
            "total_pages": math.ceil(response.count / params["page_size"]) if (response.count and params["page_size"]) else None,
        },
        status_code=status.HTTP_200_OK,
    )

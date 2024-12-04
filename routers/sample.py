from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dependencies import auth
from helpers.final_trainer import predict_user_samples, train_model_for_user
from models.sample import Event

router = APIRouter()


class BaseSample(BaseModel):
    events: List[Event]
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
async def verify(payload: VeriySamplesPayload, user=Depends(auth.extract)):
    verifying_samples = payload.samples
    result = await predict_user_samples(
        user, samples=map(lambda x: x.model_dump(), verifying_samples)
    )

    return JSONResponse(
        content=result,
        status_code=status.HTTP_201_CREATED,
    )

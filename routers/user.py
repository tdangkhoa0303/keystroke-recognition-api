from datetime import datetime, timezone
from typing import List, Optional, Any

import logging
from dependencies import auth
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from helpers.final_trainer import (
    is_model_existed,
    predict_user_samples,
    train_model_for_user,
)
from models.response import Response
from models.sample import Event, Sample
from models.session import Session
from models.user import User
from fastapi.responses import JSONResponse
from vendors.supabase import auth_admin_client, supabase

router = APIRouter()


class BaseSample(BaseModel):
    events: List[Any]
    createdAt: datetime


class SignUpPayload(BaseModel):
    email: str
    firstName: str
    lastName: str
    password: str
    phoneNumber: str
    samples: List[BaseSample]
    securityQuestionAnswer: str
    securityQuestion: str
    tp: Any


@router.post("/sign-up", status_code=status.HTTP_201_CREATED)
async def sign_up(payload: SignUpPayload):
    user = {}
    try:
        response = auth_admin_client.create_user(
            {
                "email": payload.email,
                "password": payload.password,
                "user_metadata": {
                    "last_name": payload.lastName,
                    "first_name": payload.firstName,
                },
                "email_confirm": True,
            }
        )
        user = response.user
    except Exception as error:
        logging.error(error)
        raise HTTPException(
            status_code=500,
            detail="Failed to create new user",
        )

    if payload.samples:
        response = (
            supabase.table("samples")
            .insert(
                list(
                    map(
                        lambda x: {"user_id": user.id, "events": x.events},
                        payload.samples,
                    )
                )
            )
            .execute()
        )
    # await train_model_for_user(created_user["_id"])

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={},
    )


class SignInPayload(BaseModel):
    email: str
    password: str
    samples: Optional[List[Any]]
    tp: Any


@router.post("/sign-in")
async def login(payload: SignInPayload):
    if payload.samples is None:
        raise HTTPException(detail="Bad request", status_code=400)

    user = {}
    try:
        response = supabase.auth.sign_in_with_password(
            {"email": payload.email, "password": payload.password}
        )
        user = response.user
    except Exception as error:
        logging.error(error)
        raise HTTPException(
            status_code=400,
            detail="Invalid credentials",
        )
    is_legitimate = False
    if is_model_existed(user.id):
        predict_result = await predict_user_samples(user, samples=payload.samples)
        print(predict_result)
        is_legitimate = len(list(filter(lambda x: x == 0, predict_result))) == 0

    if (is_legitimate) is False:
        raise HTTPException(detail="required_2fa", status_code=403)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"accessToken": response.session.access_token},
    )


@router.get("/me")
async def get_me(user=Depends(auth.extract)):
    user_metadata = user.user_metadata
    print(user_metadata)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=User(
            id=user.id,
            email=user.email,
            firstName=user_metadata["first_name"],
            lastName=user_metadata["last_name"],
            createdAt=user.created_at,
        ).model_dump(mode="json"),
    )


class StoreSamplesPayload(BaseModel):
    samples: List[Any]


@router.post("/samples")
async def post_samples(payload: StoreSamplesPayload, user=Depends(auth.extract)):
    if payload.samples:
        print(payload.samples[0]["events"])
        supabase.table("samples").insert(
            list(
                map(
                    lambda x: {"user_id": user.id, "events": x["events"]},
                    payload.samples,
                )
            )
        ).execute()

    return JSONResponse(
        content={"message": "Samples are stored successfully"},
        status_code=status.HTTP_201_CREATED,
    )

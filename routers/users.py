from datetime import datetime
import math
from typing import List, Any

import logging
from constants import SecurityThreshold, SessionStatus
from dependencies import auth
from fastapi import APIRouter, HTTPException, status, Depends, Request
from helpers.session import extract_session_id
from pydantic import BaseModel
from helpers.trainer import (
    is_model_existed,
    predict_user_samples,
    train_model_for_user,
)

from fastapi.responses import JSONResponse
from queries.estimator import query_latest_estimator
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
    samples: List[BaseSample]
    enableBehavioralBiometrics: bool


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
                    "security_level": SecurityThreshold.MEDIUM.name,
                    "enable_behavioural_biometrics": payload.enableBehavioralBiometrics,
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
                        lambda x: {
                            "user_id": user.id,
                            "events": x.events,
                            "predicted_score": 1,
                            "is_legitimate": True,
                            "security_level": user.user_metadata["security_level"],
                        },
                        payload.samples,
                    )
                )
            )
            .execute()
        )
        await train_model_for_user(user)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={},
    )


class SignInPayload(BaseModel):
    email: str
    password: str


@router.post("/sign-in")
async def login(payload: SignInPayload, request: Request):
    user = {}
    session = {}
    try:
        response = supabase.auth.sign_in_with_password(
            {"email": payload.email, "password": payload.password}
        )
        user = response.user
        session = response.session
    except Exception as error:
        logging.error(error)
        raise HTTPException(
            status_code=400,
            detail="Invalid credentials",
        )
    user_agent = request.headers.get("user-agent")

    user_profile = (
        supabase.table("profiles").select("*").eq("id", user.id).single().execute()
    ).data
    requied_keystroke_verification = user_profile[
        "enable_behavioural_biometrics"
    ] and is_model_existed(user.id)

    session_metadata = (
        supabase.table("session_metadata")
        .insert(
            {
                "ua": user_agent,
                "ip": request.client.host,
                "expires_at": session.expires_at,
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
                "is_revoked": False,
                "user_id": user.id,
                "status": (
                    SessionStatus.PENDING.value
                    if requied_keystroke_verification
                    else SessionStatus.ACTIVE.value
                ),
            }
        )
        .execute()
    ).data[0]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "id": session_metadata["id"],
            "accessToken": response.session.access_token,
            "refreshToken": response.session.refresh_token,
            "requiredKeystrokeVerification": requied_keystroke_verification,
        },
    )


@router.get("/me")
async def get_me(
    user=Depends(auth.extract),
    _=Depends(auth.required_keystroke_verification(limit_in_minutes=math.inf)),
):
    user_profile_response = (
        supabase.table("profiles").select("*").eq("id", user.id).single().execute()
    )
    user_profile = user_profile_response.data

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "id": user.id,
            "email": user_profile["email"],
            "firstName": user_profile["first_name"],
            "lastName": user_profile["last_name"],
            "securityLevel": user_profile["security_level"],
            "enableBehaviouralBiometrics": user_profile[
                "enable_behavioural_biometrics"
            ],
            "createdAt": str(user.created_at),
            "role": user_profile["role"],
        },
    )


class StoreSamplesPayload(BaseModel):
    samples: List[Any]


@router.post("/samples")
async def post_samples(
    payload: StoreSamplesPayload,
    request: Request,
    user=Depends(auth.extract),
    profile=Depends(auth.extract_profile),
):
    session_metadata = request.state.session_metadata
    if payload.samples and session_metadata is not None:
        supabase.table("samples").insert(
            list(
                map(
                    lambda x: {
                        "user_id": user.id,
                        "events": x["events"],
                        "predicted_score": 1,
                        "session_id": session_metadata["id"],
                        "security_level": profile["security_level"],
                        "is_legitimate": True,
                    },
                    payload.samples,
                )
            )
        ).execute()

    return JSONResponse(
        content={"message": "Samples are stored successfully"},
        status_code=status.HTTP_201_CREATED,
    )


class UpdateSecurityConfigs(BaseModel):
    security_level: str
    enable_behavioural_biometrics: bool


@router.post("/security")
async def update_security_configs(
    payload: UpdateSecurityConfigs, user=Depends(auth.extract)
):
    (
        supabase.table("profiles")
        .update(
            {
                "security_level": payload.security_level,
                "enable_behavioural_biometrics": payload.enable_behavioural_biometrics,
            }
        )
        .eq("id", user.id)
        .execute()
    )

    return JSONResponse(
        content={"message": "Update security configs successfully"},
        status_code=status.HTTP_200_OK,
    )


@router.get("/estimators/latest")
async def get_latest_estimator(user=Depends(auth.extract)):
    estimator = query_latest_estimator(user.id)

    if estimator is None:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, content={})

    return JSONResponse(
        content=estimator,
        status_code=status.HTTP_200_OK,
    )


class VeriySamplesPayload(BaseModel):
    samples: List[Any]


@router.post("/verify-session")
async def verify(
    payload: VeriySamplesPayload,
    request: Request,
    user=Depends(auth.extract),
    profile=Depends(auth.extract_profile),
):
    session_metadata = request.state.session_metadata
    is_legitimate = await predict_user_samples(
        user,
        profile=profile,
        samples=payload.samples,
        session_id=session_metadata["id"],
    )

    if is_legitimate is False:
        raise HTTPException(
            status_code=400,
            detail="Invalid credentials",
        )

    return JSONResponse(
        status_code=status.HTTP_204_NO_CONTENT,
        content={},
    )

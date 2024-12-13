from datetime import datetime
import json
from typing import List, Optional, Any

import logging
from constants import SecurityThreshold
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
    samples: Optional[List[Any]]
    tp: Any


@router.post("/sign-in")
async def login(payload: SignInPayload, request: Request):
    if payload.samples is None:
        raise HTTPException(detail="Bad request", status_code=400)

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
    session_metadata_response = (
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
            }
        )
        .execute()
    )

    session_metadata = session_metadata_response.data[0]
    user_profile = (
        supabase.table("profiles").select("*").eq("id", user.id).single().execute()
    ).data

    is_legitimate = False
    if is_model_existed(user.id):
        is_legitimate = await predict_user_samples(
            user,
            profile=user_profile,
            samples=payload.samples,
            session_id=session_metadata["id"],
        )
    else:
        is_legitimate = True

    if (is_legitimate) is False:
        supabase.table("session_metadata").update({"is_revoked": True}).eq(
            "id", session_metadata["id"]
        ).execute()

        raise HTTPException(
            status_code=400,
            detail="Invalid credentials",
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "accessToken": response.session.access_token,
            "refreshToken": response.session.refresh_token,
        },
    )


@router.get("/me")
async def get_me(user=Depends(auth.extract)):
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

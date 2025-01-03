from datetime import datetime
from fastapi import HTTPException, status, Request
from helpers.trainer import is_model_existed


async def extract(request: Request):
    try:
        user = request.state.user
        if user is None:
            raise Exception()

        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def extract_profile(request: Request):
    try:
        user_profile = request.state.user_profile
        if user_profile is None:
            raise Exception()

        return user_profile
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def required_keystroke_verification(force: bool = False, limit_in_minutes: int = 5):
    async def dependency(request: Request):
        try:
            user_profile = request.state.user_profile
            if user_profile is None:
                raise Exception()

            is_keystroke_verification_enabled = user_profile["enable_behavioural_biometrics"]
            if force and not is_keystroke_verification_enabled:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="keystroke_verification_not_enabled",
                )
                
            if not is_model_existed(user_profile["id"]) or (not is_keystroke_verification_enabled and not force):
                return True
            
            session_metadata = request.state.session_metadata
            last_mfa_verified = session_metadata["last_mfa_verified"]
            if last_mfa_verified is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="keystroke_verification_required",
                )
                
            if (datetime.now() - last_mfa_verified).total_seconds() / 60 > limit_in_minutes:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="keystroke_verification_required",
                )
            
        except Exception:
            return False
        
    return dependency
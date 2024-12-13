from fastapi import HTTPException, status, Request


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

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
        
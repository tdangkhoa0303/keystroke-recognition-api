import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi import HTTPException, Request
from vendors.supabase import supabase
from routers import samples, sessions, statistics, users

app = FastAPI()

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def required_authentication(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    token = request.headers.get("authorization", "").replace("Bearer ", "")

    if token:
        try:
            auth = supabase.auth.get_user(token)
            session_metadata = (
                supabase.table("session_metadata")
                .select("*")
                .eq("access_token", token)
                .single()
                .execute()
            ).data

            user_profile = (
                supabase.table("profiles")
                .select("*")
                .eq("id", auth.user.id)
                .single()
                .execute()
            ).data

            request.state.session_metadata = session_metadata
            request.state.user_profile = user_profile
            request.state.user = auth.user

        except Exception as e:
            logging.error(e)

    return await call_next(request)


logging.basicConfig(level=logging.INFO)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logging.error(
        f"HTTP exception: {exc.detail} - Status code: {exc.status_code} - Request: {request.url}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "reason": (
                exc.detail.model_dump() if "model_dump" in exc.detail else exc.detail
            )
        },
    )


app.include_router(users.router, prefix="/api/users")
app.include_router(samples.router, prefix="/api/samples")
app.include_router(sessions.router, prefix="/api/sessions")
app.include_router(statistics.router, prefix="/api/statistics")

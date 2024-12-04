import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi import HTTPException, Request
from vendors.supabase import auth_admin_client, supabase
from routers import user, sample

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
            request.state.user = auth.user
            supabase.postgrest.auth(token)

        except Exception as e:
            logging.error(e)

    return await call_next(request)


# Set up logging
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


app.include_router(user.router, prefix="/api/users")
app.include_router(sample.router, prefix="/api/samples")

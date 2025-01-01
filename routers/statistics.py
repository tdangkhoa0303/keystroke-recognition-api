from datetime import datetime
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse
from dependencies import auth
from vendors.supabase import supabase

router = APIRouter()


@router.get("/users")
async def get_user_stats(
    _=Depends(auth.extract_profile),
    start_date: datetime = Query(
        None, description="Filter users created on or after this date"
    ),
    end_date: datetime = Query(
        None, description="Filter users created on or before this date"
    ),
):
    query = supabase.table("profiles")
    query = query.select(
        "total_enabled:enable_behavioural_biometrics::int.sum(), total_users:id.count()",
    )

    # if start_date:
    #     query = query.gte("created_at", start_date.isoformat())
    # if end_date:
    #     query = query.lte("created_at", end_date.isoformat())

    response = query.execute()
    result = response.data[0]

    return JSONResponse(
        content={
            "totalUsers": result["total_users"],
            "enabledUsers": result["total_enabled"],
        },
        status_code=status.HTTP_200_OK,
    )


@router.get("/samples")
async def get_sample_stats(
    _=Depends(auth.extract_profile),
    start_date: datetime = Query(
        None, description="Filter users created on or after this date"
    ),
    end_date: datetime = Query(
        None, description="Filter users created on or before this date"
    ),
):
    query = supabase.table("samples")

    query = query.select(
        "total_samples:id.count(), total_success:is_legitimate::int.sum()",
    )

    if start_date:
        query = query.gte("created_at", start_date.isoformat())
    if end_date:
        query = query.lte("created_at", end_date.isoformat())

    response = query.execute()
    result = response.data[0]
    print(result)

    return JSONResponse(
        content={
            "totalSamples": result["total_samples"],
            "totalSuccess": result["total_success"],
        },
        status_code=status.HTTP_200_OK,
    )


@router.get("/verifications")
async def get_verifications_stats(
    start_date: datetime = Query(None, description="Filter data from this date"),
    end_date: datetime = Query(None, description="Filter data up to this date"),
):
    try:
        # Base query
        query = supabase.table("samples")

        # Fetch data
        query = query.select("created_at, is_legitimate", count="exact")

        # Apply date filters if provided
        if start_date:
            query = query.gte("created_at", start_date.isoformat())
        if end_date:
            query = query.lte("created_at", end_date.isoformat())

        data = query.execute()
        if not data.data:
            return JSONResponse(
                content=[],
                status_code=status.HTTP_200_OK,
            )

        # Process data to generate chart format
        chart_data = {}
        for row in data.data:
            date_key = row["created_at"][:10]  # Extract the date part
            date_key = datetime.strptime(date_key, "%Y-%m-%d").strftime("%d-%m")

            if date_key not in chart_data:
                chart_data[date_key] = {
                    "date": date_key,
                    "imposter": 0,
                    "legitimate": 0,
                }

            if row["is_legitimate"]:
                chart_data[date_key]["legitimate"] += 1
            else:
                chart_data[date_key]["imposter"] += 1

        # Convert to list
        chart_data_list = list(chart_data.values())

        return JSONResponse(
            content=chart_data_list,
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return {"error": str(e)}

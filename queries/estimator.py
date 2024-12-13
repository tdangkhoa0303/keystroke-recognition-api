from vendors.supabase import supabase


def query_latest_estimator(user_id: str):
    response = (
        supabase.table("estimators")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .single()
        .execute()
    )

    return response.data

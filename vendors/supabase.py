from configs import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL
from supabase import create_client
from supabase.lib.client_options import ClientOptions

supabase = create_client(
  SUPABASE_URL, 
  SUPABASE_SERVICE_ROLE_KEY, 
  options=ClientOptions(
    auto_refresh_token=False,
    persist_session=False,
  ))

# Access auth admin api
auth_admin_client = supabase.auth.admin
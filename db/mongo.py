import motor.motor_asyncio

from configs import MONGODB_DB_NAME, MONGODB_URL


client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = client[MONGODB_DB_NAME]

users = db["users"]
sessions = db["sessions"]
samples = db["samples"]
histories = db["histories"]

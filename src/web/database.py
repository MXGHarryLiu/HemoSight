from motor.motor_asyncio import AsyncIOMotorClient

def get_database():
    client = AsyncIOMotorClient("hematology-mongodb", 27017, 
                                directConnection=True)
    db = client['hematology']
    return db
from pymongo import MongoClient

# Hardcoded MongoDB config (adjust if needed)
MONGO_URI = "mongodb+srv://kevin:Year2006@cluster0.c40cp0n.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "hackrx"
COLLECTION_NAME = "policy_clauses"

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def fetch_all_clauses():
    """
    Fetch all clauses from MongoDB.
    Each document must have a 'clause' field.
    """
    collection = get_mongo_collection()
    return list(collection.find({}, {"_id": 0, "clause": 1}))

def save_clauses_to_mongo(clauses: list[str]):
    """
    Save a list of clause strings into MongoDB.
    """
    collection = get_mongo_collection()
    docs = [{"clause": clause} for clause in clauses]
    collection.insert_many(docs)

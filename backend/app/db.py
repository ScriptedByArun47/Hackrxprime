from pymongo import MongoClient, errors

MONGO_URI = "mongodb+srv://kevin:Year2006@cluster0.c40cp0n.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "hackrx"
COLLECTION_NAME = "policy_clauses"

def get_mongo_collection():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Ensure unique index on (source_id, clause)
        collection.create_index([("source_id", 1), ("clause", 1)], unique=True)
        return collection
    except Exception as e:
        print("❌ MongoDB connection error:", e)
        return None

def fetch_all_clauses():
    collection = get_mongo_collection()
    if collection is None:
        print("❌ No collection found.")
        return []
    return list(collection.find({}, {"_id": 0, "clause": 1}))

def save_clauses_to_mongo(clauses: list[str], source_id: str):
    collection = get_mongo_collection()
    if collection is None:
        print("❌ Collection is None. Cannot insert.")
        return
    docs = [{"clause": clause, "source_id": source_id} for clause in clauses]
    if not docs:
        print("⚠️ No clauses to insert.")
        return
    try:
        result = collection.insert_many(docs, ordered=False)
        print(f"✅ Inserted {len(result.inserted_ids)} clauses from {source_id}.")
    except errors.BulkWriteError as bwe:
        inserted_count = len([e for e in bwe.details.get('writeErrors', []) if e.get('code') != 11000])
        print(f"⚠️ Some duplicates skipped. Inserted {inserted_count} new clauses from {source_id}.")
    except Exception as e:
        print("❌ Error inserting into MongoDB:", e)

def fetch_clauses_by_source_id(source_id: str):
    collection = get_mongo_collection()
    if collection is None:
        print("❌ Collection is None. Cannot fetch.")
        return []
    return list(collection.find({"source_id": source_id}, {"_id": 0, "clause": 1}))

import json
import os
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

mongo_db_url = os.getenv("MONGO_DB_URL")
if not mongo_db_url:
    raise ValueError("MONGO_DB_URL environment variable is not set")

client = MongoClient(mongo_db_url)
DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "sensor_fault_database")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "wafer_fault")

csv_file = Path(__file__).parent / "notebooks" / "wafer_23012020_041211.csv"
df = pd.read_csv(csv_file)
df = df.drop("Unnamed: 0", axis=1, errors="ignore")

json_record = list(json.loads(df.T.to_json()).values())
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

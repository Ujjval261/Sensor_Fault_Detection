import os
from pathlib import Path


env_file = Path(__file__).resolve().parents[2] / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip().lstrip("\ufeff"), value.strip())

AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "sensor_fault_database")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "wafer_fault")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Good/Bad")
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
MODEL_FILE_NAME = os.getenv("MODEL_FILE_NAME", "model")
MODEL_FILE_EXTENSION = os.getenv("MODEL_FILE_EXTENSION", ".pkl")
MODEL_FILE_PATH = os.path.join(os.getcwd(), "artifacts", MODEL_FILE_NAME + MODEL_FILE_EXTENSION)
artifact_folder = os.path.join(os.getcwd(), "artifacts")

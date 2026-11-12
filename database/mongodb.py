
import os
from datetime import datetime, timezone
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError


load_dotenv()


@lru_cache(maxsize=1)
def get_database():
    """Connect to MongoDB Atlas and return the database."""

    mongo_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv(
        "MONGODB_DATABASE",
        "customervoice_db"
    )

    if not mongo_uri:
        raise ValueError(
            "MONGODB_URI is missing from your .env file."
        )

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000,
    )

    # Verify the connection
    client.admin.command("ping")

    database = client[database_name]

    # Create an index for unique analysis IDs
    database.analysis_runs.create_index(
        "run_id",
        unique=True
    )

    # Index recent analysis searches
    database.analysis_runs.create_index(
        [("created_at", DESCENDING)]
    )

    return database


def test_connection():
    """Check whether MongoDB is accessible."""

    try:
        database = get_database()
        database.command("ping")

        return {
            "success": True,
            "message": "MongoDB connection successful!"
        }

    except (PyMongoError, ValueError) as error:
        return {
            "success": False,
            "message": str(error)
        }


def save_analysis_run(analysis_data):
    """Save one completed CustomerVoice analysis."""

    database = get_database()

    document = {
        **analysis_data,
        "created_at": datetime.now(timezone.utc),
        "status": "completed",
    }

    database.analysis_runs.insert_one(document)

    return document["run_id"]


def get_recent_analysis_runs(limit=10):
    """Retrieve the most recent analysis runs."""

    database = get_database()

    runs = database.analysis_runs.find(
        {},
        {
            "_id": 0,
            "run_id": 1,
            "created_at": 1,
            "sample_size": 1,
            "status": 1,
            "average_rating": 1,
        }
    ).sort(
        "created_at",
        DESCENDING
    ).limit(limit)

    return list(runs)


def get_analysis_by_id(run_id):
    """Retrieve one complete analysis using its ID."""

    database = get_database()

    return database.analysis_runs.find_one(
        {"run_id": run_id},
        {"_id": 0}
    )

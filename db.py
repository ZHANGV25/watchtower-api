"""Database backend selector.

Import this instead of database/database_dynamo directly.
Selects based on WATCHTOWER_DB_BACKEND env var.
"""
import os

if os.getenv("WATCHTOWER_DB_BACKEND") == "dynamodb":
    from database_dynamo import *  # noqa: F401, F403
else:
    from database import *  # noqa: F401, F403

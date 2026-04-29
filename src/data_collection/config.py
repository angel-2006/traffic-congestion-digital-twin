import os
from dotenv import load_dotenv

load_dotenv()

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

if not TOMTOM_API_KEY:
    raise ValueError("TOMTOM_API_KEY not found in .env file")
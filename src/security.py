import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "Authorization"

# Expecting the key in the 'Authorization: Bearer <key>' header
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header_auth)):

 

    """Dependency function to validate the API key from the Authorization header."""
    if not API_KEY:
        # This case handles if the server itself is missing the API_KEY config
        # It's an internal server error, should not happen in production if configured correctly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on server."
        )

    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing"
        )

    # Expecting "Bearer <key>"
    parts = api_key_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <key>'."
        )

    provided_key = parts[1]

    if provided_key == API_KEY:
        return provided_key # Or just return True, the key itself isn't usually needed by the endpoint
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

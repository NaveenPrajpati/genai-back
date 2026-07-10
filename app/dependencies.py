from typing import Annotated
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from bson import ObjectId
import jwt

from app.database import get_db
from app.services.user_service import decode_token

_bearer = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> dict:
    try:
        payload = decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Refresh tokens are only valid at /refresh, never as a bearer access token.
    # Tokens issued before the type claim existed omit it and are treated as access.
    if payload.get("type") == "refresh":
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id or not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = await get_db()["users"].find_one(
        {"_id": ObjectId(user_id)}, {"password_hash": 0}
    )
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    # Deactivated guests are retained but no longer usable — force re-auth.
    if user.get("deactivated"):
        raise HTTPException(status_code=401, detail="Account deactivated")
    # Reject tokens issued before the last password reset (session invalidation).
    if payload.get("tv", 0) != user.get("token_version", 0):
        raise HTTPException(status_code=401, detail="Token no longer valid")
    return user


async def get_current_real_user(
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    if user.get("is_guest"):
        raise HTTPException(
            status_code=403, detail="Guest accounts cannot perform this action"
        )
    return user

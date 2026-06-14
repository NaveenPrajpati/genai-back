import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from app.models.user import UserCreate, UserLogin
from app.database import get_db
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
GUEST_TTL_HOURS = 24
USER_TOKEN_DAYS = 7


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260000)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored: str) -> bool:
    salt_hex, key_hex = stored.split(":")
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260000)
    return key.hex() == key_hex


def create_access_token(user_id: str, is_guest: bool = False) -> str:
    expiry = (
        timedelta(hours=GUEST_TTL_HOURS)
        if is_guest
        else timedelta(days=USER_TOKEN_DAYS)
    )
    payload = {
        "sub": user_id,
        "is_guest": is_guest,
        "exp": datetime.now(timezone.utc) + expiry,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def _collection():
    return get_db()["users"]


async def get_all_users() -> list[dict]:
    users = await _collection().find({}, {"password_hash": 0}).to_list(length=None)
    return users


async def get_user_by_id(user_id: str) -> dict | None:
    if not ObjectId.is_valid(user_id):
        return None
    return await _collection().find_one(
        {"_id": ObjectId(user_id)}, {"password_hash": 0}
    )


async def signup_user(user: UserCreate) -> tuple[dict, str]:
    col = _collection()
    if await col.find_one({"email": user.email}):
        raise ValueError("Email already registered")
    record = {
        "uid": str(uuid.uuid4()),
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "description": user.description,
        "password_hash": _hash_password(user.password),
        "is_guest": False,
        "diet": "vegetarian",
        "protein_target": 100,
    }
    result = await col.insert_one(record)
    created = await col.find_one({"_id": result.inserted_id}, {"password_hash": 0})

    token = create_access_token(str(result.inserted_id), is_guest=False)
    return created, token


async def login_user(credentials: UserLogin) -> tuple[dict, str]:
    col = _collection()
    user = await col.find_one({"email": credentials.email})
    if (
        not user
        or user.get("is_guest")
        or not _verify_password(credentials.password, user.get("password_hash", ""))
    ):
        raise ValueError("Invalid email or password")
    user.pop("password_hash", None)
    token = create_access_token(str(user["_id"]), is_guest=False)
    return user, token


async def create_guest_user() -> tuple[dict, str]:
    col = _collection()
    short_id = uuid.uuid4().hex[:10]
    expires_at = datetime.now(timezone.utc) + timedelta(hours=GUEST_TTL_HOURS)
    record = {
        "uid": str(uuid.uuid4()),
        "name": f"Guest_{short_id}",
        "email": f"guest_{short_id}@guest.local",
        "role": "user",
        "description": None,
        "is_guest": True,
        "diet": "vegetarian",
        "protein_target": 100,
        "expires_at": expires_at,
    }
    result = await col.insert_one(record)
    created = await col.find_one({"_id": result.inserted_id}, {"password_hash": 0})
    token = create_access_token(str(result.inserted_id), is_guest=True)
    return created, token


async def convert_guest_to_real(user_id: str, user: UserCreate) -> tuple[dict, str]:
    col = _collection()
    if not ObjectId.is_valid(user_id):
        raise ValueError("Invalid user id")
    existing = await col.find_one({"_id": ObjectId(user_id), "is_guest": True})
    if not existing:
        raise ValueError("Guest account not found")
    if await col.find_one({"email": user.email, "_id": {"$ne": ObjectId(user_id)}}):
        raise ValueError("Email already registered")
    await col.update_one(
        {"_id": ObjectId(user_id)},
        {
            "$set": {
                "name": user.name,
                "email": user.email,
                "role": user.role,
                "description": user.description,
                "password_hash": _hash_password(user.password),
                "is_guest": False,
            },
            "$unset": {"expires_at": ""},
        },
    )
    updated = await col.find_one({"_id": ObjectId(user_id)}, {"password_hash": 0})
    token = create_access_token(user_id, is_guest=False)
    return updated, token


async def update_expo_push_token(user_id: str, expo_push_token: str) -> dict | None:
    col = _collection()
    if not ObjectId.is_valid(user_id):
        return None
    result = await col.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"expo_push_token": expo_push_token}},
    )
    if result.matched_count == 0:
        return None
    return await col.find_one({"_id": ObjectId(user_id)}, {"password_hash": 0})


async def cleanup_expired_guests() -> int:
    """Failsafe cleanup — MongoDB TTL index is the primary mechanism."""
    col = _collection()
    result = await col.delete_many(
        {
            "is_guest": True,
            "expires_at": {"$lt": datetime.now(timezone.utc)},
        }
    )
    if result.deleted_count:
        print(f"[cleanup] Deleted {result.deleted_count} expired guest account(s)")
    return result.deleted_count

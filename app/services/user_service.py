import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from app.models.user import UserCreate, UserLogin
from app.database import get_db
from app.services.email_service import send_email
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
GUEST_TTL_HOURS = 24
USER_TOKEN_DAYS = 7
RESET_CODE_TTL_MINUTES = 15


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


async def request_password_reset(email: str) -> None:
    """Generate a one-time reset code and deliver it via email.

    Always returns None regardless of whether the email exists, so callers
    cannot use this endpoint to enumerate registered accounts. Guest accounts
    have no password and are ignored.
    """
    col = _collection()
    user = await col.find_one({"email": email})
    if not user or user.get("is_guest"):
        return None

    code = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=RESET_CODE_TTL_MINUTES)
    await col.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "reset_code_hash": _hash_password(code),
                "reset_code_expires_at": expires_at,
            }
        },
    )
    name = user.get("name") or "there"
    await send_email(
        to=email,
        subject="Your password reset code",
        body=(
            f"Hi {name},\n\n"
            f"Your password reset code is {code}.\n"
            f"It expires in {RESET_CODE_TTL_MINUTES} minutes.\n\n"
            "If you didn't request this, you can safely ignore this email."
        ),
    )
    return None


async def reset_password(email: str, code: str, new_password: str) -> tuple[dict, str]:
    col = _collection()
    user = await col.find_one({"email": email})
    stored = user.get("reset_code_hash") if user else None
    expires_at = user.get("reset_code_expires_at") if user else None
    # Motor returns naive datetimes (client is not tz_aware); treat them as UTC.
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if (
        not user
        or user.get("is_guest")
        or not stored
        or not expires_at
        or expires_at < datetime.now(timezone.utc)
        or not _verify_password(code, stored)
    ):
        raise ValueError("Invalid or expired reset code")

    await col.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"password_hash": _hash_password(new_password)},
            "$unset": {"reset_code_hash": "", "reset_code_expires_at": ""},
        },
    )
    updated = await col.find_one({"_id": user["_id"]}, {"password_hash": 0})
    token = create_access_token(str(user["_id"]), is_guest=False)
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

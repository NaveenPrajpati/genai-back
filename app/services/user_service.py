import hashlib
import hmac
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from app.models.user import UserCreate, UserLogin
from app.database import get_db
from app.services.email_service import send_email
import jwt

logger = logging.getLogger(__name__)

# A forgeable token secret is a full account-takeover hole, so never fall back to
# a known/placeholder value in production. In prod (APP_ENV=production) a strong
# JWT_SECRET is mandatory; in dev/test we generate an ephemeral one (consistent
# within the process) so tokens still work without committing a secret.
_PLACEHOLDER_SECRET = "change-me-in-production"
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or JWT_SECRET == _PLACEHOLDER_SECRET:
    if os.getenv("APP_ENV", "development").lower() == "production":
        raise RuntimeError(
            "JWT_SECRET must be set to a strong, unique secret in production "
            "(the insecure placeholder default is not allowed)."
        )
    logger.warning(
        "JWT_SECRET is unset or using the insecure placeholder — generating an "
        "ephemeral dev secret. Set JWT_SECRET in your environment."
    )
    JWT_SECRET = secrets.token_urlsafe(32)

JWT_ALGORITHM = "HS256"
GUEST_TTL_HOURS = 24
USER_TOKEN_DAYS = 7
REFRESH_TOKEN_DAYS = 30  # long-lived; exchanged for fresh access tokens via /refresh
RESET_CODE_TTL_MINUTES = 15
EMAIL_VERIFY_TTL_MINUTES = 30
MAX_RESET_ATTEMPTS = 5  # wrong-code guesses before the code is invalidated

PBKDF2_ITERATIONS = 600_000  # OWASP 2023 guidance for PBKDF2-HMAC-SHA256
_LEGACY_ITERATIONS = 260_000  # hashes written before the bump (salt:key, no count)


def _hash_password(password: str) -> str:
    # Format: salt_hex:iterations:key_hex — the iteration count travels with the
    # hash so it can be raised over time without breaking existing credentials.
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}:{PBKDF2_ITERATIONS}:{key.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        parts = stored.split(":")
        if len(parts) == 3:
            salt_hex, iter_str, key_hex = parts
            iterations = int(iter_str)
        elif len(parts) == 2:
            # Legacy hashes had no embedded iteration count.
            salt_hex, key_hex = parts
            iterations = _LEGACY_ITERATIONS
        else:
            return False
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), iterations)
        # Constant-time compare to avoid leaking match progress via timing.
        return hmac.compare_digest(key.hex(), key_hex)
    except (ValueError, AttributeError):
        return False


def _needs_rehash(stored: str) -> bool:
    """True if a stored hash predates the current cost (legacy 2-part format or
    fewer iterations), so it can be upgraded transparently on next login."""
    parts = stored.split(":")
    if len(parts) == 3:
        try:
            return int(parts[1]) < PBKDF2_ITERATIONS
        except ValueError:
            return True
    return True  # legacy salt:key or malformed → upgrade


# A throwaway hash so the verify path runs even when an email is unknown,
# equalizing login response time and preventing email enumeration via timing.
_DUMMY_HASH = _hash_password("timing-equalizer-not-a-real-password")


def create_access_token(
    user_id: str, is_guest: bool = False, token_version: int = 0
) -> str:
    expiry = (
        timedelta(hours=GUEST_TTL_HOURS)
        if is_guest
        else timedelta(days=USER_TOKEN_DAYS)
    )
    payload = {
        "sub": user_id,
        "is_guest": is_guest,
        "type": "access",
        # Bumped on password reset to invalidate all previously-issued tokens.
        "tv": token_version,
        "exp": datetime.now(timezone.utc) + expiry,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str, token_version: int = 0) -> str:
    """Long-lived token whose only purpose is to mint fresh access tokens at
    /refresh. Carries the same `tv` as access tokens, so a password reset or
    logout-all (which bumps token_version) invalidates it too."""
    payload = {
        "sub": user_id,
        "type": "refresh",
        "tv": token_version,
        "exp": datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _issue_tokens(
    user_id: str, is_guest: bool = False, token_version: int = 0
) -> tuple[str, str | None]:
    """Return (access_token, refresh_token). Guests get no refresh token — their
    account is deleted after GUEST_TTL_HOURS, so a 30-day refresh token would
    outlive it and is pointless."""
    access = create_access_token(user_id, is_guest=is_guest, token_version=token_version)
    refresh = None if is_guest else create_refresh_token(user_id, token_version)
    return access, refresh


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


async def signup_user(user: UserCreate) -> tuple[dict, str, str | None]:
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
        "token_version": 0,
        "email_verified": False,
        "diet": "vegetarian",
        "protein_target": 100,
    }
    result = await col.insert_one(record)
    # Send a verification code (best-effort; never blocks account creation).
    await _issue_email_verification(result.inserted_id, user.email, user.name)
    created = await col.find_one({"_id": result.inserted_id}, {"password_hash": 0})

    access, refresh = _issue_tokens(str(result.inserted_id), is_guest=False)
    return created, access, refresh


async def login_user(credentials: UserLogin) -> tuple[dict, str, str | None]:
    col = _collection()
    user = await col.find_one({"email": credentials.email})
    # Always run a hash verify (against a dummy hash when the user is absent) so
    # the response time doesn't reveal whether the email exists.
    stored = user.get("password_hash", "") if user else _DUMMY_HASH
    password_ok = _verify_password(credentials.password, stored or _DUMMY_HASH)
    if not user or user.get("is_guest") or not password_ok:
        raise ValueError("Invalid email or password")
    # Transparently upgrade old/low-cost hashes now that we have the plaintext.
    if _needs_rehash(stored):
        await col.update_one(
            {"_id": user["_id"]},
            {"$set": {"password_hash": _hash_password(credentials.password)}},
        )
    user.pop("password_hash", None)
    access, refresh = _issue_tokens(
        str(user["_id"]), is_guest=False, token_version=user.get("token_version", 0)
    )
    return user, access, refresh


async def create_guest_user() -> tuple[dict, str, str | None]:
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
    access, refresh = _issue_tokens(str(result.inserted_id), is_guest=True)
    return created, access, refresh


async def convert_guest_to_real(
    user_id: str, user: UserCreate
) -> tuple[dict, str, str | None]:
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
                "token_version": 0,
                "email_verified": False,
            },
            "$unset": {"expires_at": ""},
        },
    )
    # The newly-attached email is unverified — send a code.
    await _issue_email_verification(ObjectId(user_id), user.email, user.name)
    updated = await col.find_one({"_id": ObjectId(user_id)}, {"password_hash": 0})
    access, refresh = _issue_tokens(user_id, is_guest=False, token_version=0)
    return updated, access, refresh


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
                "reset_code_attempts": 0,
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


async def reset_password(
    email: str, code: str, new_password: str
) -> tuple[dict, str, str | None]:
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
    ):
        raise ValueError("Invalid or expired reset code")

    if not _verify_password(code, stored):
        # Count the failed guess; after MAX_RESET_ATTEMPTS, burn the code so a
        # 6-digit code can't be brute-forced within its TTL.
        attempts = (user.get("reset_code_attempts") or 0) + 1
        if attempts >= MAX_RESET_ATTEMPTS:
            await col.update_one(
                {"_id": user["_id"]},
                {"$unset": {
                    "reset_code_hash": "",
                    "reset_code_expires_at": "",
                    "reset_code_attempts": "",
                }},
            )
        else:
            await col.update_one(
                {"_id": user["_id"]}, {"$set": {"reset_code_attempts": attempts}}
            )
        raise ValueError("Invalid or expired reset code")

    # Success: rotate the password, invalidate the code, and bump token_version
    # so any previously-issued tokens (a possibly-compromised session) stop working.
    new_tv = (user.get("token_version") or 0) + 1
    await col.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"password_hash": _hash_password(new_password), "token_version": new_tv},
            "$unset": {
                "reset_code_hash": "",
                "reset_code_expires_at": "",
                "reset_code_attempts": "",
            },
        },
    )
    updated = await col.find_one({"_id": user["_id"]}, {"password_hash": 0})
    access, refresh = _issue_tokens(str(user["_id"]), is_guest=False, token_version=new_tv)
    return updated, access, refresh


async def _issue_email_verification(user_id, email: str, name: str | None) -> None:
    """Generate a one-time email-verification code, store it (hashed, with TTL +
    attempt counter), and email it. Best-effort — never raises, so it can't break
    signup/conversion even if email delivery fails."""
    try:
        code = f"{secrets.randbelow(1_000_000):06d}"
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=EMAIL_VERIFY_TTL_MINUTES)
        await _collection().update_one(
            {"_id": user_id},
            {"$set": {
                "email_verify_code_hash": _hash_password(code),
                "email_verify_expires_at": expires_at,
                "email_verify_attempts": 0,
            }},
        )
        await send_email(
            to=email,
            subject="Verify your email",
            body=(
                f"Hi {name or 'there'},\n\n"
                f"Your email verification code is {code}.\n"
                f"It expires in {EMAIL_VERIFY_TTL_MINUTES} minutes.\n"
            ),
        )
    except Exception as e:
        logger.error("issue email verification error: %s", e)


async def verify_email(email: str, code: str) -> dict:
    """Validate the verification code and mark the email verified. Mirrors the
    reset-code flow: TTL + attempt counter so the code can't be brute-forced."""
    col = _collection()
    user = await col.find_one({"email": email})
    if user and user.get("email_verified"):
        # Already verified — treat as success (idempotent).
        user.pop("password_hash", None)
        return user

    stored = user.get("email_verify_code_hash") if user else None
    expires_at = user.get("email_verify_expires_at") if user else None
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if (
        not user
        or not stored
        or not expires_at
        or expires_at < datetime.now(timezone.utc)
    ):
        raise ValueError("Invalid or expired verification code")

    if not _verify_password(code, stored):
        attempts = (user.get("email_verify_attempts") or 0) + 1
        if attempts >= MAX_RESET_ATTEMPTS:
            await col.update_one(
                {"_id": user["_id"]},
                {"$unset": {
                    "email_verify_code_hash": "",
                    "email_verify_expires_at": "",
                    "email_verify_attempts": "",
                }},
            )
        else:
            await col.update_one(
                {"_id": user["_id"]}, {"$set": {"email_verify_attempts": attempts}}
            )
        raise ValueError("Invalid or expired verification code")

    await col.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"email_verified": True},
            "$unset": {
                "email_verify_code_hash": "",
                "email_verify_expires_at": "",
                "email_verify_attempts": "",
            },
        },
    )
    updated = await col.find_one({"_id": user["_id"]}, {"password_hash": 0})
    return updated


async def resend_verification(email: str) -> None:
    """Reissue a verification code. Non-enumerating: always returns None, and
    no-ops for unknown, guest, or already-verified accounts."""
    user = await _collection().find_one({"email": email})
    if not user or user.get("is_guest") or user.get("email_verified"):
        return None
    await _issue_email_verification(user["_id"], email, user.get("name"))
    return None


async def refresh_access_token(refresh_token: str) -> tuple[dict, str, str]:
    """Exchange a valid refresh token for a fresh access + refresh token pair
    (rotation). Raises ValueError on any invalid/expired/revoked token.

    Revocation piggybacks on token_version: a password reset or logout-all bumps
    it, so refresh tokens minted before then fail the `tv` check here."""
    try:
        payload = decode_token(refresh_token)
    except jwt.ExpiredSignatureError:
        raise ValueError("Refresh token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid refresh token")

    if payload.get("type") != "refresh":
        raise ValueError("Invalid refresh token")

    user_id = payload.get("sub")
    if not user_id or not ObjectId.is_valid(user_id):
        raise ValueError("Invalid refresh token")

    user = await _collection().find_one(
        {"_id": ObjectId(user_id)}, {"password_hash": 0}
    )
    if not user or user.get("is_guest"):
        raise ValueError("Invalid refresh token")
    if payload.get("tv", 0) != user.get("token_version", 0):
        raise ValueError("Refresh token no longer valid")

    access, refresh = _issue_tokens(
        user_id, is_guest=False, token_version=user.get("token_version", 0)
    )
    return user, access, refresh


async def bump_token_version(user_id: str) -> None:
    """Invalidate every token previously issued to this user (logout-all)."""
    if not ObjectId.is_valid(user_id):
        return
    await _collection().update_one(
        {"_id": ObjectId(user_id)}, {"$inc": {"token_version": 1}}
    )


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

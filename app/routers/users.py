from typing import Annotated
from fastapi import APIRouter, HTTPException, Depends, Request
from app.models.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    AuthResponse,
    ExpoPushTokenUpdate,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    VerifyEmailRequest,
    ResendVerificationRequest,
)
from app.services import user_service
from app.services import rate_limit
from app.dependencies import get_current_user

router = APIRouter()


@router.post("/signup", response_model=AuthResponse, status_code=201)
async def signup(user: UserCreate, request: Request):
    await rate_limit.limit_ip(request, "signup", limit=10, window_seconds=3600)
    try:
        created, token = await user_service.signup_user(user)
        return {
            "message": "Account created successfully",
            "token": token,
            "user": created,
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(credentials: UserLogin, request: Request):
    # Throttle by IP (broad) and by email (per-account brute-force defense).
    await rate_limit.limit_ip(request, "login", limit=20, window_seconds=300)
    await rate_limit.limit_key("login", credentials.email, limit=5, window_seconds=900)
    try:
        user, token = await user_service.login_user(credentials)
        return {"message": "Login successful", "token": token, "user": user}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/forgot-password", status_code=202)
async def forgot_password(payload: ForgotPasswordRequest, request: Request):
    """Send a one-time password-reset code to the account (via push).

    Always returns the same response whether or not the email is registered,
    to avoid leaking which accounts exist.
    """
    await rate_limit.limit_ip(request, "forgot", limit=5, window_seconds=900)
    await rate_limit.limit_key("forgot", payload.email, limit=3, window_seconds=900)
    await user_service.request_password_reset(payload.email)
    return {
        "message": "If an account exists for that email, a reset code has been sent."
    }


@router.post("/reset-password", response_model=AuthResponse)
async def reset_password(payload: ResetPasswordRequest, request: Request):
    # Cap code-guessing attempts per IP and per account (defense-in-depth on top
    # of the per-code attempt counter in the service).
    await rate_limit.limit_ip(request, "reset", limit=10, window_seconds=900)
    await rate_limit.limit_key("reset", payload.email, limit=5, window_seconds=900)
    try:
        user, token = await user_service.reset_password(
            payload.email, payload.code, payload.new_password
        )
        return {"message": "Password reset successful", "token": token, "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/verify-email", response_model=UserResponse)
async def verify_email(payload: VerifyEmailRequest, request: Request):
    """Confirm ownership of the signup email with the one-time code."""
    await rate_limit.limit_ip(request, "verify", limit=10, window_seconds=900)
    await rate_limit.limit_key("verify", payload.email, limit=5, window_seconds=900)
    try:
        return await user_service.verify_email(payload.email, payload.code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/resend-verification", status_code=202)
async def resend_verification(payload: ResendVerificationRequest, request: Request):
    """Reissue a verification code. Non-enumerating: same response regardless of
    whether the email exists or is already verified."""
    await rate_limit.limit_ip(request, "resend", limit=5, window_seconds=900)
    await rate_limit.limit_key("resend", payload.email, limit=3, window_seconds=900)
    await user_service.resend_verification(payload.email)
    return {"message": "If the email needs verification, a new code has been sent."}


@router.post("/guest", response_model=AuthResponse, status_code=201)
async def create_guest(request: Request):
    """Create a temporary guest account valid for 24 hours."""
    await rate_limit.limit_ip(request, "guest", limit=20, window_seconds=3600)
    created, token = await user_service.create_guest_user()
    return {"message": "Guest account created", "token": token, "user": created}


@router.post("/convert-guest", response_model=AuthResponse)
async def convert_guest(
    user: UserCreate,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Convert the authenticated guest account into a permanent account."""
    if not current_user.get("is_guest"):
        raise HTTPException(
            status_code=400, detail="Only guest accounts can be converted"
        )
    try:
        updated, token = await user_service.convert_guest_to_real(
            str(current_user["_id"]), user
        )
        return {
            "message": "Account converted successfully",
            "token": token,
            "user": updated,
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/logout")
async def logout(current_user: Annotated[dict, Depends(get_current_user)]):
    """Invalidate all of this user's existing tokens (logout on all devices)."""
    await user_service.bump_token_version(str(current_user["_id"]))
    return {"message": "Logged out on all devices"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: Annotated[dict, Depends(get_current_user)]):
    return current_user


@router.patch("/me/expo-push-token", response_model=UserResponse)
async def update_expo_push_token(
    payload: ExpoPushTokenUpdate,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Register or update the Expo push notification token for the current user."""
    updated = await user_service.update_expo_push_token(
        str(current_user["_id"]), payload.expo_push_token
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated


@router.get("/", response_model=list[UserResponse])
async def get_users(current_user: Annotated[dict, Depends(get_current_user)]):
    # Listing all users is an admin-only operation — never public.
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return await user_service.get_all_users()


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    # A user may fetch only their own record; admins may fetch anyone.
    if str(current_user["_id"]) != user_id and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view this user")
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

from typing import Annotated
from fastapi import APIRouter, HTTPException, Depends
from app.models.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    AuthResponse,
    ExpoPushTokenUpdate,
)
from app.services import user_service
from app.dependencies import get_current_user

router = APIRouter()


@router.post("/signup", response_model=AuthResponse, status_code=201)
async def signup(user: UserCreate):
    try:
        created, token = await user_service.signup_user(user)
        return {"message": "Account created successfully", "token": token, "user": created}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(credentials: UserLogin):
    try:
        user, token = await user_service.login_user(credentials)
        return {"message": "Login successful", "token": token, "user": user}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/guest", response_model=AuthResponse, status_code=201)
async def create_guest():
    """Create a temporary guest account valid for 24 hours."""
    created, token = await user_service.create_guest_user()
    return {"message": "Guest account created", "token": token, "user": created}


@router.post("/convert-guest", response_model=AuthResponse)
async def convert_guest(
    user: UserCreate,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """Convert the authenticated guest account into a permanent account."""
    if not current_user.get("is_guest"):
        raise HTTPException(status_code=400, detail="Only guest accounts can be converted")
    try:
        updated, token = await user_service.convert_guest_to_real(
            str(current_user["_id"]), user
        )
        return {"message": "Account converted successfully", "token": token, "user": updated}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


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
async def get_users():
    return await user_service.get_all_users()


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

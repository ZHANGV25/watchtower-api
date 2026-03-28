"""Auth REST endpoints: register, login, me."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
from auth import create_token, hash_password, verify_password
from middleware import require_auth
from models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str


@router.post("/register", response_model=AuthResponse)
async def register(req: AuthRequest):
    existing = await db.get_user_by_username(req.username)
    if existing:
        raise HTTPException(400, "Username already taken")

    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
    )
    await db.create_user(user)
    token = create_token(user.id, user.username)
    return AuthResponse(token=token, user_id=user.id, username=user.username)


@router.post("/login", response_model=AuthResponse)
async def login(req: AuthRequest):
    user = await db.get_user_by_username(req.username)
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")

    token = create_token(user.id, user.username)
    return AuthResponse(token=token, user_id=user.id, username=user.username)


@router.get("/me")
async def me(user: dict = Depends(require_auth)):
    return {"user_id": user["user_id"], "username": user["username"]}

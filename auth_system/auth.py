"""
auth.py - JWT Authentication for Spacecraft Health AI
"""
import os, secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY   = os.environ.get("ILSA_SECRET", secrets.token_hex(32))
ALGORITHM    = "HS256"
EXPIRE_HOURS = 24 * 7   # 7 days

security = HTTPBearer(auto_error=False)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def get_username_from_token(token: str) -> Optional[str]:
    p = decode_token(token)
    return p.get("sub") if p else None

def get_current_user_optional(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    cookie: Optional[str] = Cookie(default=None, alias="ilsa_token"),
) -> Optional[str]:
    token = (creds.credentials if creds else None) or cookie
    return get_username_from_token(token) if token else None

def get_current_user_required(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    cookie: Optional[str] = Cookie(default=None, alias="ilsa_token"),
) -> str:
    token = (creds.credentials if creds else None) or cookie
    username = get_username_from_token(token) if token else None
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return username

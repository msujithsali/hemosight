"""JWT auth (python-jose).

Author: M Sujith Sali, ISE Dept, VTU Karnataka.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

ALGO = "HS256"
SECRET = os.environ.get("HEMOSIGHT_JWT_SECRET", "change-me-in-production")


def create_token(subject: str, minutes: int = 60) -> str:
    payload = {"sub": subject, "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)}
    return jwt.encode(payload, SECRET, algorithm=ALGO)


def decode_token(token: str) -> str | None:
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGO]).get("sub")
    except JWTError:
        return None

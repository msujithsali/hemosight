"""Local artifact encryption (AES-256-GCM) + argon2 password hashing.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.

On-device model weights and cached samples are encrypted at rest with
AES-256-GCM; passwords are hashed with argon2id. Keys are read from the
environment / a mounted secret, never hard-coded.
"""
from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt(plaintext: bytes, key: bytes | None = None) -> bytes:
    key = key or os.urandom(32)
    if len(key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    nonce = os.urandom(12)
    return key + nonce + AESGCM(key).encrypt(nonce, plaintext, None)


def decrypt(blob: bytes) -> bytes:
    key, nonce, ct = blob[:32], blob[32:44], blob[44:]
    return AESGCM(key).decrypt(nonce, ct, None)


def hash_password(password: str) -> str:
    from argon2 import PasswordHasher

    return PasswordHasher().hash(password)


def verify_password(hashed: str, password: str) -> bool:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError

    try:
        return PasswordHasher().verify(hashed, password)
    except VerifyMismatchError:
        return False

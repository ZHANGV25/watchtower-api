"""Tests for auth module."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth import create_token, hash_password, verify_password, verify_token


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed)

    def test_wrong_password(self):
        hashed = hash_password("correct")
        assert not verify_password("wrong", hashed)

    def test_different_hashes(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        # Different salts = different hashes
        assert h1 != h2
        # But both verify
        assert verify_password("same", h1)
        assert verify_password("same", h2)

    def test_invalid_hash_format(self):
        assert not verify_password("pass", "nocolon")
        assert not verify_password("pass", "")


class TestTokens:
    def test_create_and_verify(self):
        token = create_token("user123", "testuser")
        payload = verify_token(token)
        assert payload is not None
        assert payload["user_id"] == "user123"
        assert payload["username"] == "testuser"

    def test_invalid_token(self):
        assert verify_token("garbage") is None
        assert verify_token("") is None
        assert verify_token("a.b.c") is None

    def test_tampered_token(self):
        token = create_token("user1", "test")
        # Tamper with signature
        parts = token.split(".")
        tampered = parts[0] + ".tampered"
        assert verify_token(tampered) is None

    def test_expired_token(self):
        import auth
        old_expiry = auth.TOKEN_EXPIRY_HOURS
        auth.TOKEN_EXPIRY_HOURS = -1  # Already expired
        token = create_token("user1", "test")
        auth.TOKEN_EXPIRY_HOURS = old_expiry
        assert verify_token(token) is None

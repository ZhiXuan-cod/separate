# utils/auth.py
import base64
import hashlib
import hmac
import os

import streamlit as st


def hash_password(password: str, iterations: int = 100_000) -> str:
    """Hash a password using PBKDF2-SHA256 with a random salt."""
    salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return (
        f"pbkdf2_sha256${iterations}$"
        f"{base64.b64encode(salt).decode('utf-8')}$"
        f"{base64.b64encode(pwd_hash).decode('utf-8')}"
    )


def verify_password(plain_password: str, stored_password: str) -> bool:
    """
    Verify a plain-text password against a stored PBKDF2 hash.
    Uses constant-time comparison to prevent timing attacks.

    Only accepts properly-hashed PBKDF2 passwords.
    Plaintext fallback has been removed for security — any password that is
    not in the expected format is rejected outright.
    """
    if not stored_password:
        return False

    if not stored_password.startswith("pbkdf2_sha256$"):
        # Reject legacy or malformed entries — never compare in the clear.
        return False

    try:
        _, iterations_str, salt_b64, hash_b64 = stored_password.split("$", 3)
        iterations = int(iterations_str)
        salt = base64.b64decode(salt_b64.encode("utf-8"))
        expected_hash = base64.b64decode(hash_b64.encode("utf-8"))
        candidate = hashlib.pbkdf2_hmac(
            "sha256", plain_password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(candidate, expected_hash)
    except Exception:
        return False


def register_user(email: str, password: str, name: str) -> tuple[bool, str]:
    """
    Register a new user in Supabase.
    Returns (success: bool, message: str).
    """
    if st.session_state.get("supabase") is None:
        return False, "Supabase not connected. Please check your configuration."
    try:
        response = (
            st.session_state.supabase
            .table("users")
            .select("*")
            .eq("email", email)
            .execute()
        )
        if response.data:
            return False, "This email address is already registered."

        st.session_state.supabase.table("users").insert({
            "email": email,
            "name": name,
            "password": hash_password(password),
        }).execute()
        return True, "Registration successful. Please log in."
    except Exception as exc:
        return False, f"Registration failed: {exc}"


def authenticate_user(
    email: str, password: str
) -> tuple[bool, str | None, str | None]:
    """
    Authenticate a user by email and password.
    Returns (success: bool, name: str | None, email: str | None).
    """
    if st.session_state.get("supabase") is None:
        return False, None, None
    try:
        response = (
            st.session_state.supabase
            .table("users")
            .select("*")
            .eq("email", email)
            .execute()
        )
        if not response.data:
            return False, None, None

        user = response.data[0]
        if verify_password(password, user.get("password", "")):
            return True, user["name"], user["email"]
        return False, None, None
    except Exception as exc:
        st.error(f"Authentication error: {exc}")
        return False, None, None

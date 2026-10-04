import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Identity:
    key_id: str


class KeyAuth:
    """Only SHA-256 key digests are kept in the credentials file."""

    def __init__(self, path: str, demo: bool = False):
        self.keys: list[tuple[str, str]] = []
        if demo:
            raw = os.getenv("GATEWAY_DEMO_KEY", "demo-key-change-me")
            self.keys = [("demo", hashlib.sha256(raw.encode()).hexdigest())]
        else:
            records = json.loads(Path(path).read_text(encoding="utf-8"))
            for record in records:
                digest = record["sha256"]
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid API key SHA-256 digest")
                if record.get("enabled", True):
                    self.keys.append((record["id"], digest))
            if not self.keys or len({key[0] for key in self.keys}) != len(self.keys):
                raise ValueError("at least one key and unique active key IDs required")

    def authenticate(self, authorization: str | None) -> Identity | None:
        scheme, _, value = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not value:
            return None
        digest = hashlib.sha256(value.encode()).hexdigest()
        identity = None
        for key_id, expected in self.keys:
            if hmac.compare_digest(digest, expected):
                identity = Identity(key_id)
        return identity

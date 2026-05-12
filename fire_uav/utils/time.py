from __future__ import annotations

from datetime import datetime, timezone


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso_z(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["utc_now", "utc_iso_z"]
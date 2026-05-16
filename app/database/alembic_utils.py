"""Alembic helpers."""


def make_sync_database_url(database_url: str) -> str:
    """Convert async SQLAlchemy URLs to sync URLs for Alembic."""
    replacements = {
        "sqlite+aiosqlite://": "sqlite://",
        "postgresql+asyncpg://": "postgresql://",
        "postgresql+psycopg://": "postgresql://",
    }
    for async_prefix, sync_prefix in replacements.items():
        if database_url.startswith(async_prefix):
            return sync_prefix + database_url[len(async_prefix) :]
    return database_url

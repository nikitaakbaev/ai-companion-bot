"""Database engine and session helpers."""

import logging

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database import models  # noqa: F401

logger = logging.getLogger(__name__)


def create_engine_from_url(database_url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine from a database URL."""
    if database_url.endswith(":memory:"):
        return create_async_engine(database_url, future=True, poolclass=StaticPool)
    return create_async_engine(database_url, future=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine, auto_create_tables: bool) -> None:
    """Create database tables when explicitly enabled for development/tests."""
    if not auto_create_tables:
        logger.info("AUTO_CREATE_TABLES=false. Skipping Base.metadata.create_all.")
        return

    logger.info("AUTO_CREATE_TABLES=true. Creating database tables if they do not exist")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

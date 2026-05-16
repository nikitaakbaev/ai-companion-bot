import pytest

from app.database.session import create_engine_from_url, create_session_factory, init_db


@pytest.fixture
async def session_factory():
    engine = create_engine_from_url("sqlite+aiosqlite:///:memory:")
    await init_db(engine, auto_create_tables=True)
    factory = create_session_factory(engine)
    try:
        yield factory
    finally:
        await engine.dispose()

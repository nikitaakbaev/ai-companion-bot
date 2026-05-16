from app.database.alembic_utils import make_sync_database_url


def test_make_sync_database_url_for_sqlite_aiosqlite() -> None:
    assert (
        make_sync_database_url("sqlite+aiosqlite:///./data/bot.db")
        == "sqlite:///./data/bot.db"
    )


def test_make_sync_database_url_leaves_sync_url_unchanged() -> None:
    assert make_sync_database_url("sqlite:///./data/bot.db") == "sqlite:///./data/bot.db"

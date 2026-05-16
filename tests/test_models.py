from app.database.base import Base
from app.database.models import (
    AgentAction,
    AgentState,
    BotSettings,
    Chat,
    DiaryEntry,
    MediaFile,
    Message,
    User,
)


def test_stage_5_models_are_registered() -> None:
    table_names = set(Base.metadata.tables)

    assert User.__tablename__ in table_names
    assert Chat.__tablename__ in table_names
    assert Message.__tablename__ in table_names
    assert AgentAction.__tablename__ in table_names
    assert BotSettings.__tablename__ in table_names
    assert DiaryEntry.__tablename__ in table_names
    assert MediaFile.__tablename__ in table_names
    assert AgentState.__tablename__ in table_names


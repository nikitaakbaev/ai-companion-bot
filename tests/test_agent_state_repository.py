from app.database.repositories import (
    get_or_create_agent_state,
    get_or_create_user,
    update_agent_state,
)


async def test_get_or_create_agent_state_creates_default(session_factory) -> None:
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        state = await get_or_create_agent_state(session, user.id)

        assert state.last_emotion == "neutral"
        assert state.user_id == user.id


async def test_update_agent_state(session_factory) -> None:
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        state = await update_agent_state(
            session,
            user.id,
            last_emotion="happy",
            last_action_type="send_message",
        )

        assert state.last_emotion == "happy"
        assert state.last_action_type == "send_message"


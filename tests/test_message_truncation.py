from app.database.repositories import (
    get_or_create_chat,
    get_or_create_user,
    prepare_message_text,
    save_message,
)


def test_prepare_message_text_truncates_and_sets_metadata() -> None:
    text, metadata = prepare_message_text("abcdef", 3)

    assert text == "abc"
    assert metadata == {"truncated": True, "original_length": 6}


async def test_save_message_truncates_long_text(session_factory) -> None:
    async with session_factory() as session:
        user = await get_or_create_user(session, 123, None, None, None)
        chat = await get_or_create_chat(session, 456, user.id, None, "private")
        message = await save_message(
            session=session,
            chat_id=chat.id,
            user_id=user.id,
            role="user",
            text="abcdef",
            message_type="text",
            telegram_message_id=1,
            max_stored_message_length=3,
        )

        assert message.text == "abc"
        assert message.metadata_json == {"truncated": True, "original_length": 6}

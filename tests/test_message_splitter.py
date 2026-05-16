from app.agent.message_splitter import split_into_messages


def test_short_text_returns_single_message() -> None:
    assert split_into_messages("hello", max_length=100) == ["hello"]


def test_long_text_splits_into_multiple_messages() -> None:
    text = "First sentence. Second sentence. Third sentence."

    messages = split_into_messages(text, max_length=20)

    assert len(messages) > 1
    assert " ".join(messages) == text


def test_empty_messages_are_not_returned() -> None:
    assert split_into_messages("\n\n   \n") == []


"""initial full schema."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial_full_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("telegram_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(length=255), nullable=True),
        sa.Column("first_name", sa.String(length=255), nullable=True),
        sa.Column("last_name", sa.String(length=255), nullable=True),
        sa.Column("language_code", sa.String(length=32), nullable=True),
        sa.Column("is_bot", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_telegram_id"), "users", ["telegram_id"], unique=True)

    op.create_table(
        "chats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("telegram_chat_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("chat_type", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("telegram_chat_id", "user_id", name="uq_chat_telegram_user"),
    )
    op.create_index(op.f("ix_chats_telegram_chat_id"), "chats", ["telegram_chat_id"], unique=False)

    op.create_table(
        "agent_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("last_emotion", sa.String(length=64), nullable=False),
        sa.Column("last_action_type", sa.String(length=64), nullable=True),
        sa.Column("last_interaction_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_proactive_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("short_term_summary", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_agent_states_user_id"), "agent_states", ["user_id"], unique=True)

    op.create_table(
        "bot_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("character_name", sa.String(length=255), nullable=False),
        sa.Column("character_description", sa.Text(), nullable=False),
        sa.Column("personality_style", sa.Text(), nullable=False),
        sa.Column("llm_model", sa.String(length=255), nullable=True),
        sa.Column("vision_model", sa.String(length=255), nullable=True),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("proactive_enabled", sa.Boolean(), nullable=False),
        sa.Column("proactive_min_interval_minutes", sa.Integer(), nullable=False),
        sa.Column("proactive_max_interval_minutes", sa.Integer(), nullable=False),
        sa.Column("timezone", sa.String(length=128), nullable=False),
        sa.Column("silent_hours_start", sa.String(length=16), nullable=False),
        sa.Column("silent_hours_end", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_bot_settings_user_id"), "bot_settings", ["user_id"], unique=True)

    op.create_table(
        "diary_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("facts_about_user", sa.JSON(), nullable=True),
        sa.Column("facts_about_relationship", sa.JSON(), nullable=True),
        sa.Column("topics", sa.JSON(), nullable=True),
        sa.Column("importance", sa.Integer(), nullable=False),
        sa.Column("emotion", sa.String(length=64), nullable=True),
        sa.Column("source_date", sa.Date(), nullable=True),
        sa.Column("embedding_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_diary_entries_user_id"), "diary_entries", ["user_id"], unique=False)
    op.create_index(op.f("ix_diary_entries_created_at"), "diary_entries", ["created_at"], unique=False)

    op.create_table(
        "media_files",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("chat_id", sa.Integer(), nullable=True),
        sa.Column("telegram_file_id", sa.String(length=512), nullable=True),
        sa.Column("telegram_file_unique_id", sa.String(length=512), nullable=True),
        sa.Column("file_type", sa.String(length=64), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("local_path", sa.Text(), nullable=True),
        sa.Column("original_file_name", sa.String(length=512), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_media_files_user_id"), "media_files", ["user_id"], unique=False)
    op.create_index(op.f("ix_media_files_chat_id"), "media_files", ["chat_id"], unique=False)
    op.create_index(op.f("ix_media_files_created_at"), "media_files", ["created_at"], unique=False)

    op.create_table(
        "agent_actions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("chat_id", sa.Integer(), nullable=True),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("input_json", sa.JSON(), nullable=True),
        sa.Column("output_json", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_agent_actions_user_id"), "agent_actions", ["user_id"], unique=False)
    op.create_index(op.f("ix_agent_actions_chat_id"), "agent_actions", ["chat_id"], unique=False)
    op.create_index(op.f("ix_agent_actions_action_type"), "agent_actions", ["action_type"], unique=False)
    op.create_index(op.f("ix_agent_actions_status"), "agent_actions", ["status"], unique=False)
    op.create_index(op.f("ix_agent_actions_created_at"), "agent_actions", ["created_at"], unique=False)

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("message_type", sa.String(length=32), nullable=False),
        sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
        sa.Column("reply_to_message_id", sa.BigInteger(), nullable=True),
        sa.Column("media_file_id", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"]),
        sa.ForeignKeyConstraint(["media_file_id"], ["media_files.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_chat_id"), "messages", ["chat_id"], unique=False)
    op.create_index(op.f("ix_messages_user_id"), "messages", ["user_id"], unique=False)
    op.create_index(op.f("ix_messages_role"), "messages", ["role"], unique=False)
    op.create_index(op.f("ix_messages_message_type"), "messages", ["message_type"], unique=False)
    op.create_index(op.f("ix_messages_created_at"), "messages", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_messages_created_at"), table_name="messages")
    op.drop_index(op.f("ix_messages_message_type"), table_name="messages")
    op.drop_index(op.f("ix_messages_role"), table_name="messages")
    op.drop_index(op.f("ix_messages_user_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_chat_id"), table_name="messages")
    op.drop_table("messages")
    op.drop_index(op.f("ix_agent_actions_created_at"), table_name="agent_actions")
    op.drop_index(op.f("ix_agent_actions_status"), table_name="agent_actions")
    op.drop_index(op.f("ix_agent_actions_action_type"), table_name="agent_actions")
    op.drop_index(op.f("ix_agent_actions_chat_id"), table_name="agent_actions")
    op.drop_index(op.f("ix_agent_actions_user_id"), table_name="agent_actions")
    op.drop_table("agent_actions")
    op.drop_index(op.f("ix_media_files_created_at"), table_name="media_files")
    op.drop_index(op.f("ix_media_files_chat_id"), table_name="media_files")
    op.drop_index(op.f("ix_media_files_user_id"), table_name="media_files")
    op.drop_table("media_files")
    op.drop_index(op.f("ix_diary_entries_created_at"), table_name="diary_entries")
    op.drop_index(op.f("ix_diary_entries_user_id"), table_name="diary_entries")
    op.drop_table("diary_entries")
    op.drop_index(op.f("ix_bot_settings_user_id"), table_name="bot_settings")
    op.drop_table("bot_settings")
    op.drop_index(op.f("ix_agent_states_user_id"), table_name="agent_states")
    op.drop_table("agent_states")
    op.drop_index(op.f("ix_chats_telegram_chat_id"), table_name="chats")
    op.drop_table("chats")
    op.drop_index(op.f("ix_users_telegram_id"), table_name="users")
    op.drop_table("users")

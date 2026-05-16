# AI Companion Bot

## Что это

Python-проект Telegram AI companion bot с памятью, дневником, RAG, проактивностью, vision и генерацией изображений.

## Текущий статус

Этап 2: Telegram MVP. LLM-запросы, память, RAG, vision, ComfyUI и scheduler пока не реализованы.

## Этап 2: Telegram MVP

На этом этапе реализовано:

- запуск Telegram polling при наличии `TELEGRAM_BOT_TOKEN`;
- команды `/start`, `/help`, `/status`;
- сохранение пользователей;
- сохранение чатов;
- сохранение сообщений;
- SQLite-база через SQLAlchemy async;
- автоматическое создание таблиц при запуске.

## Возможности будущих этапов

- Telegram-бот
- LLM через OpenAI-compatible API
- Долговременная память
- Дневник
- RAG
- Проактивные сообщения
- Vision
- Генерация изображений через ComfyUI

## Установка

Требуется Python 3.11+ и `uv`.

```bash
cp .env.example .env
uv sync
```

## Настройка .env

Можно оставить `TELEGRAM_BOT_TOKEN` пустым. Приложение создаст таблицы, выведет warning и завершится корректно.

Для запуска Telegram-бота вставьте токен от BotFather:

```env
TELEGRAM_BOT_TOKEN=123456:ABC...
```

Основные параметры:

- `APP_NAME`
- `APP_ENV`
- `LOG_LEVEL`
- `TELEGRAM_BOT_TOKEN`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `DATABASE_URL`

## Запуск

```bash
uv run python -m app.main
```

Без токена приложение загрузит настройки, создаст таблицы и завершится без ошибки. С токеном оно запустит Telegram polling.

Для проверки в Telegram:

1. Откройте своего бота.
2. Отправьте `/start`.
3. Отправьте любое текстовое сообщение.
4. Проверьте, что создан файл `data/bot.db`.

## Тесты

```bash
uv run ruff check .
uv run pytest
```

## Структура проекта

```text
app/
  bot/                 Telegram handlers, router, keyboards, middlewares
  agent/               Agent loop, prompts, schemas, tools, behavior
  llm/                 OpenAI-compatible LLM, vision, embeddings
  memory/              Diary, summarizer, RAG, vector store
  database/            SQLAlchemy base, session, models, repositories
  scheduler/           Jobs, proactive loop, daily reflection
  image_generation/    ComfyUI client and prompt helpers
  services/            Telegram, typing, files, external context
  utils/               Time, JSON, retry helpers
tests/
data/
alembic/
```

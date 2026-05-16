# AI Companion Bot

## Что это

Python-проект Telegram AI companion bot с памятью, дневником, RAG, проактивностью, vision и генерацией изображений.

## Текущий статус

Этап 1: каркас проекта. Реальный Telegram polling, LLM-запросы, база данных, RAG, vision, ComfyUI и scheduler пока не реализованы.

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

На этапе 1 можно оставить `TELEGRAM_BOT_TOKEN` пустым. Приложение выведет warning и завершится корректно.

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

Ожидаемый результат: приложение загрузит настройки, настроит логирование, выведет параметры окружения и завершит stage 1 bootstrap.

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


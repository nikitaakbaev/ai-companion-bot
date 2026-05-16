# AI Companion Bot

## Что это

Python-проект Telegram AI companion bot с памятью, дневником, RAG, проактивностью, vision и генерацией изображений.

## Текущий статус

Этап 5: полноценная база данных, настройки и история. Реальный дневник, RAG, vision, ComfyUI и scheduler пока не реализованы.

## Этап 2: Telegram MVP

На этом этапе реализовано:

- запуск Telegram polling при наличии `TELEGRAM_BOT_TOKEN`;
- команды `/start`, `/help`, `/status`;
- сохранение пользователей;
- сохранение чатов;
- сохранение сообщений;
- SQLite-база через SQLAlchemy async;
- автоматическое создание таблиц при запуске.

## Этап 3: LLM-интеграция

На этом этапе бот умеет отправлять текстовые сообщения пользователя в LLM через OpenAI-compatible API и возвращать ответ модели в Telegram.

Реализовано:

- OpenAI-compatible LLM client;
- поддержка LM Studio, Ollama OpenAI-compatible endpoint и OpenAI API;
- retry для временных сетевых ошибок и 5xx;
- сборка простого prompt с историей последних сообщений;
- команда `/llm_test`;
- сохранение ответов LLM в SQLite.

## Этап 4: JSON Agent Loop и Tool Calling

На этом этапе бот больше не использует обычный текстовый ответ LLM для диалога. Теперь LLM возвращает JSON-решение, а Python валидирует его и выполняет выбранное действие через `ToolExecutor`.

Пример JSON:

```json
{
  "thought": "Пользователь поздоровался.",
  "action": "send_message",
  "messages": ["Привет.", "Я на связи."],
  "tool_input": {},
  "emotion": "happy",
  "delay_seconds": 1
}
```

Работают tools:

- `send_message`
- `ignore`

Заглушки будущих этапов:

- `remember`
- `read_diary`
- `sleep`
- `take_photo`
- `analyze_image`

Команда проверки:

```text
/agent_test
```

Если после обновления появилась ошибка с таблицей `agent_actions`, можно удалить старую SQLite-базу `data/bot.db` и запустить проект заново. В следующих этапах будет добавлен Alembic.

## Этап 5: База данных, настройки и история

На этом этапе добавлены:

- Alembic;
- полноценная схема базы;
- настройки пользователя;
- состояние агента;
- дневниковые записи;
- медиафайлы;
- история сообщений;
- история действий агента.

Новые команды:

- `/settings` — показать настройки персонажа и моделей
- `/history` — последние сообщения
- `/actions` — последние действия агента
- `/diary` — дневник памяти, пока пустой до этапа 6

### Миграции

По умолчанию `AUTO_CREATE_TABLES=false`, поэтому перед запуском нужно выполнить миграции:

```bash
mkdir data
alembic upgrade head
python -m app.main
```

Для разработки можно включить старое поведение:

```env
AUTO_CREATE_TABLES=true
```

Если используется старая SQLite-база из этапов 1-4, первая миграция может конфликтовать с уже существующими таблицами. Для этапа 5 допустимо удалить `data/bot.db` и выполнить:

```bash
alembic upgrade head
python -m app.main
```

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
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `DATABASE_URL`

### LM Studio

1. Откройте LM Studio.
2. Скачайте модель.
3. Запустите Local Server.
4. Проверьте доступность:

```bash
curl http://127.0.0.1:1234/v1/models
```

`.env`:

```env
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=имя_модели_из_/v1/models
```

### Ollama

Запустите Ollama и скачайте модель:

```bash
ollama pull qwen2.5:14b
curl http://127.0.0.1:11434/v1/models
```

`.env`:

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:14b
```

## Запуск

```bash
uv run python -m app.main
```

Без токена приложение загрузит настройки, создаст таблицы и завершится без ошибки. С токеном оно запустит Telegram polling.

Для проверки в Telegram:

1. Откройте своего бота.
2. Отправьте `/start`.
3. Отправьте `/status`.
4. Отправьте `/llm_test`.
5. Отправьте `/agent_test`.
6. Отправьте `/settings`.
7. Отправьте любое текстовое сообщение.
8. Отправьте `/history`.
9. Отправьте `/actions`.
10. Отправьте `/diary`.
11. Проверьте, что создан файл `data/bot.db`.

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

# AI Companion Bot

## Что это

Python-проект Telegram AI companion bot с памятью, дневником, RAG, проактивностью, vision и генерацией изображений.

## Текущий статус

Этап 6: дневник и долговременная память без RAG. RAG, embeddings, vision, ComfyUI и scheduler пока не реализованы.

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

## Этап 6: Дневник и долговременная память

На этом этапе бот умеет сжимать историю переписки в дневниковые записи.

Команды:

- `/sleep` — создать дневниковую сводку
- `/diary` — показать краткий дневник
- `/diary_full` — показать последние записи полностью

Пример `.env`:

```env
DIARY_ENABLED=true
DIARY_LOOKBACK_HOURS=24
DIARY_MIN_MESSAGES=3
DIARY_MAX_MESSAGES=100
DIARY_MAX_INPUT_CHARS=20000
DIARY_MAX_ENTRIES_PER_RUN=8
DIARY_SKIP_IF_EXISTS_FOR_DATE=true
```

Дневник пока не используется автоматически при ответах. На этапе 7 будет добавлен RAG: бот начнёт искать релевантные записи дневника и подмешивать их в prompt.

Для разработки можно включить старое поведение:

```env
AUTO_CREATE_TABLES=true
```

## Stage 7: RAG memory

The bot now uses long-term diary memory during normal replies.

How it works:

1. `/sleep` or automatic reflection creates diary entries.
2. Each diary entry is converted into an embedding.
3. The embedding is stored in Chroma.
4. On each new text message, the bot searches similar diary entries.
5. Relevant memories are added to the agent prompt.

Commands:

- `/memory` - show recent diary entries and embedding status.
- `/memory_search <query>` - search long-term memory.
- `/memory_reindex` - rebuild embeddings for diary entries.

Settings:

```env
RAG_ENABLED=true
RAG_TOP_K=5
RAG_MIN_SCORE=0.65
RAG_MAX_CONTEXT_CHARS=4000
EMBEDDING_PROVIDER=auto
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=./data/chroma
VECTOR_COLLECTION_NAME=diary_entries
```

`EMBEDDING_PROVIDER=auto` uses the built-in hash embedding provider. This avoids local
`torch`/`transformers` version conflicts and keeps RAG available. Set
`EMBEDDING_PROVIDER=sentence_transformers` only when the local ML dependency stack is healthy.

## Stage 8: Tool calling

Implemented tools:

- `send_message` - sends one or more Telegram messages with typing simulation.
- `ignore` - stores the action without replying.
- `sleep` - creates diary reflection entries and indexes them.
- `remember` - saves one explicit memory into diary/vector memory.
- `read_diary` - returns recent diary entries as tool output.

`take_photo` and `analyze_image` return a clear `unavailable` status until image generation and
vision are configured.

## Stage 9: ComfyUI image generation

The bot can generate a photo through ComfyUI when the agent chooses `take_photo` or when you run
`/photo_test <description>`.

Settings:

```env
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_CHECKPOINT=model.safetensors
COMFYUI_CLIP_SKIP=-2
COMFYUI_SAMPLER=euler
COMFYUI_SCHEDULER=normal
COMFYUI_STEPS=25
COMFYUI_CFG=7.0
COMFYUI_WIDTH=768
COMFYUI_HEIGHT=1024
IMAGE_GENERATION_ENABLED=true
IMAGE_BASE_POSITIVE_PROMPT=masterpiece, best quality, 1girl, long blue hair, blue eyes, pale skin, black hoodie, (boobs:0.4), dark cyber room, night
IMAGE_BASE_NEGATIVE_PROMPT=bad quality,worst quality,worst detail,sketch,censor
GENERATED_IMAGES_DIR=./data/generated_images
```

The LLM only adds contextual details such as pose, mood, camera angle, or situation. The fixed
character identity stays in `IMAGE_BASE_POSITIVE_PROMPT`.

## Stable Waifu Telegram provider

ComfyUI stays available, but image generation can also be delegated to the Telegram bot
`@StableWaifuBot` through a Telethon userbot session. This uses a normal Telegram user account,
not Telegram Bot API, because bots cannot reliably control other bots.

Install dependencies:

```bash
uv sync
```

Provider settings:

```env
IMAGE_GENERATION_ENABLED=true
IMAGE_GENERATION_PROVIDER=stable_waifu_telegram

STABLE_WAIFU_ENABLED=true
STABLE_WAIFU_BOT_USERNAME=StableWaifuBot
STABLE_WAIFU_MODEL=SkyVXL
STABLE_WAIFU_PRESET=Opus
STABLE_WAIFU_TIMEOUT_SECONDS=240
STABLE_WAIFU_POLL_INTERVAL_SECONDS=3
STABLE_WAIFU_CLICK_START_BUTTON=true
STABLE_WAIFU_CLICK_SAVE_BUTTON=true
STABLE_WAIFU_OUTPUT_DIR=./data/generated_images/stable_waifu
STABLE_WAIFU_ORIENTATION=landscape
STABLE_WAIFU_ASPECT_RATIO=16:9
STABLE_WAIFU_MODEL_SEARCH_MAX_PAGE_SWITCHES=10
STABLE_WAIFU_LANDSCAPE_KEYWORDS=Ландшафт,Landscape
STABLE_WAIFU_PORTRAIT_KEYWORDS=Портрет,Portrait
STABLE_WAIFU_MODEL_MENU_KEYWORDS=Модель,Model
STABLE_WAIFU_PRESET_MENU_KEYWORDS=Пресет,Preset
STABLE_WAIFU_ASPECT_RATIO_KEYWORDS=Соотношение,Aspect,Ratio
STABLE_WAIFU_NEXT_PAGE_KEYWORDS=>>,››,➡,Next
STABLE_WAIFU_PREVIOUS_PAGE_KEYWORDS=<<,‹‹,⬅,Back
STABLE_WAIFU_START_KEYWORDS=Начать,Start,🎉
STABLE_WAIFU_SAVE_KEYWORDS=Сохранить,Save
STABLE_WAIFU_EDITOR_WAIT_KEYWORDS=Новый арт,Проверь параметры,Модель,Пресет
STABLE_WAIFU_DEFAULT_LANDSCAPE_RATIO=16:9
STABLE_WAIFU_DEFAULT_PORTRAIT_RATIO=9:16

TELEGRAM_USERBOT_API_ID=123456
TELEGRAM_USERBOT_API_HASH=your_api_hash
TELEGRAM_USERBOT_SESSION_PATH=./data/userbot/stable_waifu.session

IMAGE_BASE_TAGS=1girl, pale skin, black hoodie, oversized hoodie, blue eyes, long dark blue hair, blue gradient hair, bangs, blue nails, soft smile, cozy dark room, night, cyber aesthetic, anime, masterpiece, best quality
IMAGE_NEGATIVE_TAGS=low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, missing fingers, extra limbs, deformed body, mutated hands, poorly drawn face, ugly, text, watermark, logo, cropped, jpeg artifacts, duplicate, distorted eyes, cross-eyed, bad proportions, messy background, low detail
IMAGE_PROMPT_MAX_LENGTH=900
STABLE_WAIFU_USE_PONY_PREFIXES=false
STABLE_WAIFU_NSFW_LEVEL=0
STABLE_WAIFU_MAX_PROMPT_TAGS=35
```

Get `TELEGRAM_USERBOT_API_ID` and `TELEGRAM_USERBOT_API_HASH` from
`https://my.telegram.org/apps`. On the first `python -m app.main` run, Telethon asks for the
phone number, login code, and 2FA password if enabled. The session is saved to
`data/userbot/stable_waifu.session`; do not commit it.

Stable Waifu diagnostics:

- `/stable_waifu_settings` - selected provider, model, preset, timeout.
- `/stable_waifu_models` - supported models and presets.
- `/stable_waifu_login_status` - Telethon configuration and session status.
- `/stable_waifu_test` - sends a test prompt, waits for the generated image, downloads it,
  sends it to the chat, and stores media metadata.

Supported models: `NekoXL`, `KanaXL`, `BlueXL`, `VersaXL`, `EpicXL`, `SkyVXL`.
Supported presets: `Modern`, `Opus`, `Eclair`.

### Stable Waifu prompting guide

Stable Waifu prompts are built by `app/image_generation/stable_waifu_prompting/`.
The LLM only provides scene-level tags, while the app injects the fixed character
identity from `IMAGE_BASE_TAGS`.

Prompt rules:

- use compact lowercase anime tags;
- use comma-separated tags only;
- avoid prose and full sentences;
- keep base identity in `IMAGE_BASE_TAGS`, not in LLM scene tags;
- use `IMAGE_NEGATIVE_TAGS` for negative prompt tags;
- use `STABLE_WAIFU_USE_PONY_PREFIXES=true` for Pony-style score prefixes;
- use `STABLE_WAIFU_NSFW_LEVEL` to choose the configured intensity level.

Supported archetypes live in
`app/image_generation/stable_waifu_prompting/scene_archetypes.py`, including:

- `cozy_selfie`
- `gaming_setup`
- `rainy_window`
- `bedroom_sleepy`
- `cafe_evening`
- `city_night`

Mood presets live in `mood_presets.py`, camera tags in `camera_tags.py`, lighting
tags in `lighting_tags.py`, and forbidden tags in `forbidden_tags.py`.

NSFW tags are not stored in `.env`. Edit
`app/image_generation/stable_waifu_prompting/nsfw_levels.py` and set the active
level with:

```env
STABLE_WAIFU_NSFW_LEVEL=0
```

Use `0` to disable NSFW additions. Higher levels append the tags configured in
`NSFW_LEVELS`.

Orientation behavior:

- `portrait` or `9:16` favors `selfie`, `close-up`, `upper body`;
- `landscape` or `16:9` favors wider room/environment composition;
- `1:1` adds centered framing.

Good scene tags:

```text
selfie, cozy room, sleepy, soft lighting
gaming setup, headphones, monitor glow, playful smile, neon lighting
rainy window, city lights, melancholic mood, soft shadows
```

Bad scene prompt:

```text
A beautiful anime girl sitting near a window while softly smiling...
```

Switch back to local ComfyUI with:

```env
IMAGE_GENERATION_PROVIDER=comfyui
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

"""Application entrypoint."""

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError

from app.agent.orchestrator import AgentOrchestrator
from app.agent.tool_executor import ToolExecutor
from app.bot.router import router
from app.config import get_settings
from app.database.session import create_engine_from_url, create_session_factory, init_db
from app.image_generation.comfyui_client import ComfyUIClient
from app.image_generation.generation_queue import ImageGenerationQueue
from app.image_generation.prompt_builder import StableWaifuPromptBuilder
from app.image_generation.service import ComfyUIImageProvider, ImageGenerationService
from app.image_generation.stable_waifu_provider import StableWaifuTelegramProvider
from app.image_generation.telegram_userbot import TelegramUserbotClient
from app.llm.embeddings import HashEmbeddingService, SentenceTransformerEmbeddingService
from app.llm.openai_compatible import OpenAICompatibleLLMClient
from app.llm.vision import OpenAICompatibleVisionClient
from app.logging_config import setup_logging
from app.memory.diary import DiaryService
from app.memory.rag import MemoryIndexer, MemoryRetriever
from app.memory.summarizer import DiarySummarizer
from app.memory.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def create_embedding_service(settings):
    """Create the configured embedding service with a dependency-free fallback."""
    fallback = HashEmbeddingService(dimensions=settings.embedding_dimensions)
    provider = settings.embedding_provider.strip().lower()
    if provider in {"auto", "hash", "local_hash", "fallback"}:
        logger.info("Using hash embedding provider")
        return fallback
    if provider in {"sentence_transformers", "sentence-transformers"}:
        logger.info(
            "Using sentence-transformers embedding provider with hash fallback",
            extra={"embedding_model": settings.embedding_model},
        )
        return SentenceTransformerEmbeddingService(
            model_name=settings.embedding_model,
            fallback_service=fallback,
        )
    logger.warning("Unknown embedding provider '%s'; using hash fallback", settings.embedding_provider)
    return fallback


async def create_image_generation_service(settings):
    """Create the configured image generation service and optional userbot."""
    if not settings.image_generation_enabled:
        return None, None

    provider_name = settings.image_generation_provider.strip().lower()
    prompt_builder = StableWaifuPromptBuilder(
        base_tags=settings.image_base_tags,
        negative_tags=settings.image_negative_tags,
        max_length=settings.image_prompt_max_length,
        orientation=settings.stable_waifu_orientation,
        aspect_ratio=settings.stable_waifu_aspect_ratio,
        use_pony_prefixes=settings.stable_waifu_use_pony_prefixes,
        nsfw_level=settings.stable_waifu_nsfw_level,
        max_tags=settings.stable_waifu_max_prompt_tags,
    )
    if provider_name == "stable_waifu_telegram":
        if not settings.stable_waifu_enabled:
            logger.warning("Stable Waifu provider selected but STABLE_WAIFU_ENABLED=false")
            return None, None
        if settings.telegram_userbot_api_id is None or not settings.telegram_userbot_api_hash:
            logger.warning("Stable Waifu provider is missing Telethon credentials")
            return None, None
        userbot = TelegramUserbotClient(
            api_id=settings.telegram_userbot_api_id,
            api_hash=settings.telegram_userbot_api_hash,
            session_path=settings.telegram_userbot_session_path,
        )
        await userbot.start()
        provider = StableWaifuTelegramProvider(
            userbot=userbot,
            queue=ImageGenerationQueue(),
            bot_username=settings.stable_waifu_bot_username,
            output_dir=settings.stable_waifu_output_dir,
            timeout_seconds=settings.stable_waifu_timeout_seconds,
            poll_interval_seconds=settings.stable_waifu_poll_interval_seconds,
            model_name=settings.stable_waifu_model,
            preset_name=settings.stable_waifu_preset,
            orientation=settings.stable_waifu_orientation,
            aspect_ratio=settings.stable_waifu_aspect_ratio,
            model_search_max_page_switches=settings.stable_waifu_model_search_max_page_switches,
            landscape_keywords=settings.stable_waifu_landscape_keywords,
            portrait_keywords=settings.stable_waifu_portrait_keywords,
            model_menu_keywords=settings.stable_waifu_model_menu_keywords,
            preset_menu_keywords=settings.stable_waifu_preset_menu_keywords,
            aspect_ratio_keywords=settings.stable_waifu_aspect_ratio_keywords,
            next_page_keywords=settings.stable_waifu_next_page_keywords,
            previous_page_keywords=settings.stable_waifu_previous_page_keywords,
            start_keywords=settings.stable_waifu_start_keywords,
            save_keywords=settings.stable_waifu_save_keywords,
            editor_wait_keywords=settings.stable_waifu_editor_wait_keywords,
            default_landscape_ratio=settings.stable_waifu_default_landscape_ratio,
            default_portrait_ratio=settings.stable_waifu_default_portrait_ratio,
            click_start_button=settings.stable_waifu_click_start_button,
            click_save_button=settings.stable_waifu_click_save_button,
        )
        return ImageGenerationService(
            provider_name="stable_waifu_telegram",
            provider=provider,
            comfy_base_positive_prompt=settings.image_base_positive_prompt,
            comfy_base_negative_prompt=settings.image_base_negative_prompt,
            stable_prompt_builder=prompt_builder,
        ), userbot

    comfy_client = ComfyUIClient(
        base_url=settings.comfyui_base_url,
        output_dir=settings.generated_images_dir,
        checkpoint=settings.comfyui_checkpoint,
        clip_skip=settings.comfyui_clip_skip,
        sampler=settings.comfyui_sampler,
        scheduler=settings.comfyui_scheduler,
        steps=settings.comfyui_steps,
        cfg=settings.comfyui_cfg,
        width=settings.comfyui_width,
        height=settings.comfyui_height,
        timeout_seconds=settings.comfyui_timeout_seconds,
        poll_interval_seconds=settings.comfyui_poll_interval_seconds,
        poll_attempts=settings.comfyui_poll_attempts,
    )
    return ImageGenerationService(
        provider_name="comfyui",
        provider=ComfyUIImageProvider(comfy_client),
        comfy_base_positive_prompt=settings.image_base_positive_prompt,
        comfy_base_negative_prompt=settings.image_base_negative_prompt,
        stable_prompt_builder=prompt_builder,
    ), None


async def main() -> None:
    """Start the application."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting %s", settings.app_name)
    logger.info("Environment: %s", settings.app_env)
    logger.info("Debug: %s", settings.debug)
    logger.info("Database URL: %s", settings.database_url)
    logger.info("LLM base URL: %s", settings.llm_base_url)
    logger.info("LLM model: %s", settings.llm_model)
    logger.info("LLM backend: OpenAI-compatible API")

    engine = create_engine_from_url(settings.database_url)
    session_factory = create_session_factory(engine)
    await init_db(engine, settings.auto_create_tables)

    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN is not set. Telegram bot is not started.")
        await engine.dispose()
        return

    llm_client = OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
        disable_thinking=settings.llm_disable_thinking,
    )
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        max_context_messages=settings.agent_context_messages,
        temperature=settings.agent_temperature,
        max_tokens=settings.agent_max_tokens,
        user_prompt_mode=settings.agent_user_prompt_mode,
        response_format_enabled=settings.agent_response_format,
    )
    diary_model = settings.diary_reflection_model or settings.llm_model
    diary_llm_client = (
        llm_client
        if diary_model == settings.llm_model
        else OpenAICompatibleLLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=diary_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_temperature=0.2,
            default_max_tokens=settings.diary_max_tokens,
            disable_thinking=settings.llm_disable_thinking,
        )
    )
    diary_summarizer = DiarySummarizer(
        llm_client=diary_llm_client,
        max_entries_per_run=settings.diary_max_entries_per_run,
        max_input_chars=settings.diary_max_input_chars,
        max_tokens=settings.diary_max_tokens,
        user_prompt_mode=settings.diary_user_prompt_mode,
    )
    memory_indexer = None
    memory_retriever = None
    if settings.rag_enabled:
        embedding_service = create_embedding_service(settings)
        vector_store = ChromaVectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.vector_collection_name,
        )
        memory_indexer = MemoryIndexer(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        memory_retriever = MemoryRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
            max_context_chars=settings.rag_max_context_chars,
        )
    diary_service = DiaryService(
        summarizer=diary_summarizer,
        min_messages=settings.diary_min_messages,
        max_messages=settings.diary_max_messages,
        lookback_hours=settings.diary_lookback_hours,
        skip_if_exists_for_date=settings.diary_skip_if_exists_for_date,
        memory_indexer=memory_indexer,
    )

    bot = Bot(token=settings.telegram_bot_token, session=AiohttpSession(timeout=120))
    vision_client = None
    if settings.vision_enabled:
        vision_client = OpenAICompatibleVisionClient(
            base_url=settings.vision_base_url or settings.llm_base_url,
            api_key=settings.vision_api_key or settings.llm_api_key,
            model=settings.vision_model,
            timeout_seconds=settings.vision_timeout_seconds,
            max_tokens=settings.vision_max_tokens,
        )
    image_service, telegram_userbot = await create_image_generation_service(settings)
    tool_executor = ToolExecutor(
        bot=bot,
        diary_service=diary_service,
        image_service=image_service,
        max_delay_seconds=settings.agent_max_delay_seconds,
        typing_seconds=settings.agent_typing_seconds,
    )
    dp = Dispatcher()
    dp.include_router(router)
    dp["session_factory"] = session_factory
    dp["orchestrator"] = orchestrator
    dp["tool_executor"] = tool_executor
    dp["diary_service"] = diary_service
    dp["memory_indexer"] = memory_indexer
    dp["memory_retriever"] = memory_retriever
    dp["vision_client"] = vision_client
    dp["image_service"] = image_service
    dp["telegram_userbot"] = telegram_userbot
    dp["settings"] = settings

    logger.info("Starting Telegram polling")
    try:
        await dp.start_polling(bot)
    except TelegramNetworkError:
        logger.exception(
            "Cannot connect to Telegram API. Check network, VPN/proxy, firewall, or try again later."
        )
    finally:
        if telegram_userbot is not None:
            await telegram_userbot.stop()
        await bot.session.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

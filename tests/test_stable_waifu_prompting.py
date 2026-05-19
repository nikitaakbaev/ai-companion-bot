from app.image_generation.stable_waifu_prompting.mood_presets import MOOD_PRESETS
from app.image_generation.stable_waifu_prompting.nsfw_levels import NSFW_LEVELS
from app.image_generation.stable_waifu_prompting.prompt_builder import StableWaifuPromptBuilder
from app.image_generation.stable_waifu_prompting.scene_archetypes import SCENE_ARCHETYPES
from app.image_generation.stable_waifu_prompting.tag_cleaner import PromptCleaner
from app.image_generation.stable_waifu_prompting.validation import PromptValidator


def test_prompt_cleaner_normalizes_commas_and_lowercase() -> None:
    cleaner = PromptCleaner()

    assert cleaner.clean(" Selfie,, Cozy Room ; Soft Lighting. ") == (
        "selfie, cozy room, soft lighting"
    )


def test_validator_removes_duplicates_forbidden_and_conflicts() -> None:
    validator = PromptValidator(max_tags=20)

    tags = validator.validate(["selfie", "selfie", "mirror selfie", "chibi", "night", "day"])

    assert tags == ["mirror selfie", "day"]


def test_validator_adds_orientation_hint() -> None:
    validator = PromptValidator(max_tags=20, orientation="portrait", aspect_ratio="9:16")

    tags = validator.validate(["cozy room"])

    assert "selfie" in tags


def test_validator_adds_aspect_ratio_hint() -> None:
    validator = PromptValidator(max_tags=20, aspect_ratio="16:9")

    tags = validator.validate(["cozy room"])

    assert "wide shot" in tags


def test_prompt_builder_uses_archetype_and_mood_presets() -> None:
    builder = StableWaifuPromptBuilder(
        base_tags="1girl, blue eyes",
        negative_tags="bad hands",
        max_length=500,
    )

    prompt = builder.build_prompt(
        {
            "archetype": "gaming_setup",
            "emotion_tags": MOOD_PRESETS["playful"],
        }
    )

    assert "gaming setup" in prompt.positive
    assert "monitor glow" in prompt.positive
    assert "playful smile" in prompt.positive


def test_prompt_builder_can_use_pony_prefixes() -> None:
    builder = StableWaifuPromptBuilder(
        base_tags="1girl",
        negative_tags="",
        max_length=500,
        use_pony_prefixes=True,
    )

    prompt = builder.build_prompt({"scene_tags": ["selfie"]})

    assert prompt.positive.startswith("score_9, score_8_up, score_7_up")


def test_prompt_builder_applies_nsfw_intensity_progression() -> None:
    low = StableWaifuPromptBuilder(
        base_tags="1girl",
        negative_tags="",
        max_length=500,
        nsfw_level=1,
    ).build_prompt({"scene_tags": "selfie"})
    high = StableWaifuPromptBuilder(
        base_tags="1girl",
        negative_tags="",
        max_length=500,
        nsfw_level=3,
    ).build_prompt({"scene_tags": "selfie"})

    assert all(tag in high.positive for tag in NSFW_LEVELS[1])
    assert len(high.positive.split(", ")) > len(low.positive.split(", "))


def test_scene_archetypes_are_available() -> None:
    assert "cozy_selfie" in SCENE_ARCHETYPES
    assert "selfie" in SCENE_ARCHETYPES["cozy_selfie"]

"""Coherent emotional tag presets for Stable Waifu prompts."""

MOOD_PRESETS = {
    "sleepy": ["sleepy", "tired eyes", "messy hair", "relaxed expression"],
    "playful": ["playful smile", "teasing smile", "peace sign"],
    "melancholic": ["melancholic mood", "sad smile", "soft shadows"],
    "shy": ["shy", "blush", "averted eyes", "soft smile"],
    "happy": ["happy", "smile", "bright expression"],
    "sad": ["sad", "teary eyes", "looking away"],
}

CONFLICTING_MOODS = (
    {"sleepy", "happy", "sad"},
    {"playful smile", "sad smile"},
    {"teasing smile", "melancholic mood"},
)

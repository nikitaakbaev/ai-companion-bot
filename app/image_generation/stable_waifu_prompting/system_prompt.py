"""System prompt for LLM-generated Stable Waifu scene tags."""

STABLE_WAIFU_SYSTEM_PROMPT = """
Generate compact Stable Waifu compatible anime tags.

Rules:
- output only comma-separated tags
- no prose
- no explanations
- no markdown
- no full sentences
- do not describe base appearance
- focus only on:
  - scene
  - emotion
  - environment
  - pose
  - camera
  - lighting

Good:
selfie, cozy room, sleepy, soft lighting

Bad:
A beautiful anime girl sitting near a window...
""".strip()

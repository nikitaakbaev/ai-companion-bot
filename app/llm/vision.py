"""Vision model placeholders."""


class VisionClient:
    """Analyzes images with a vision model."""

    async def describe_image(self, image_path: str) -> str:
        """Describe an image in later stages."""
        raise NotImplementedError


"""ComfyUI client placeholders."""


class ComfyUIClient:
    """Generates images through ComfyUI."""

    async def generate_image(self) -> str:
        """Generate an image in later stages."""
        raise NotImplementedError


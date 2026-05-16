"""File service placeholders."""

from pathlib import Path


class FileService:
    """Handles local files and downloaded media."""

    def ensure_directory(self, path: str) -> Path:
        """Create a directory if it does not exist."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory


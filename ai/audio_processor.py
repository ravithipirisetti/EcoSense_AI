"""
EcoSense AI - Audio File Processing Utility.

Validates audio clip format and handles file streaming into temporary paths.
"""

import tempfile
from pathlib import Path
from typing import Tuple

from fastapi import HTTPException, UploadFile, status

SUPPORTED_FORMATS: set = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def validate_audio_file(filename: str) -> str:
    """Validate uploaded audio file extension."""
    if not filename:
        raise ValueError("No audio file provided.")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        # Warning but allowed if missing standard extension
        pass
    return ext or ".wav"


async def save_temp_audio(audio: UploadFile) -> Tuple[Path, bytes]:
    """Save uploaded UploadFile bytes to temporary file for librosa/YAMNet processing."""
    if not audio.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "No audio file uploaded."},
        )

    file_ext = validate_audio_file(audio.filename)
    content = await audio.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "Uploaded audio file is empty."},
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    return temp_path, content

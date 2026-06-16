#!/usr/bin/env python3

import mimetypes
import os
from datetime import datetime
from typing import Dict

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import app_state as state


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def is_image_path(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_IMAGE_EXTENSIONS


def safe_image_path(filename: str) -> str:
    if not is_image_path(filename):
        raise ValueError("Unsupported image type")

    path = state.agent.safe_path(filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(filename)

    return path


def image_message_data(filename: str, label: str | None = None) -> Dict[str, str]:
    path = safe_image_path(filename)
    rel = os.path.relpath(path, state.agent.working_dir).replace("\\", "/")
    mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    return {
        "filename": rel,
        "label": label or os.path.basename(rel),
        "url": f"/workdir-image/{rel}",
        "mime_type": mime_type,
    }


def save_uploaded_image(file: FileStorage) -> Dict[str, str]:
    original_name = secure_filename(file.filename or "upload")
    _, ext = os.path.splitext(original_name)
    ext = ext.lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {original_name}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"{stamp}-{original_name}"
    rel = os.path.join("uploads", filename)
    path = state.agent.safe_path(rel)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    file.save(path)

    return image_message_data(rel, original_name)

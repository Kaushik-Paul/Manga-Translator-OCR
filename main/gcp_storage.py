"""
GCP Cloud Storage integration for the Manga Translator.

Handles authentication, listing folders/images, downloading,
uploading translated images, and generating presigned download URLs.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import zipfile
from datetime import timedelta
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET_NAME = "manga-ocr-translation"
_RAW_PREFIX = "raw-manga"
_TRANSLATED_PREFIX = "translated-manga"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _get_credentials() -> service_account.Credentials:
    """Decode GCP_SERVICE_ACCOUNT_BASE64 env var into credentials."""
    b64 = os.getenv("GCP_SERVICE_ACCOUNT_BASE64", "")
    if not b64:
        raise ValueError(
            "GCP_SERVICE_ACCOUNT_BASE64 is not set. "
            "Please set it in your .env file."
        )
    try:
        sa_json = base64.b64decode(b64)
        sa_info = json.loads(sa_json)
    except Exception as e:
        raise ValueError(f"Failed to decode GCP_SERVICE_ACCOUNT_BASE64: {e}") from e
    return service_account.Credentials.from_service_account_info(sa_info)


def _get_bucket_name() -> str:
    """Return the configured bucket name or the default."""
    return os.getenv("GCP_BUCKET_NAME", "").strip() or _DEFAULT_BUCKET_NAME


def get_gcs_client() -> storage.Client:
    """Create an authenticated GCS client."""
    credentials = _get_credentials()
    return storage.Client(credentials=credentials, project=credentials.project_id)


def get_bucket() -> storage.Bucket:
    """Get the configured GCS bucket."""
    client = get_gcs_client()
    bucket_name = _get_bucket_name()
    return client.bucket(bucket_name)


def list_manga_folders() -> list[str]:
    """List all manga folder names under raw-manga/."""
    client = get_gcs_client()
    bucket_name = _get_bucket_name()
    bucket = client.bucket(bucket_name)

    prefix = f"{_RAW_PREFIX}/"
    # Use delimiter to get "subdirectories"
    blobs = client.list_blobs(bucket, prefix=prefix, delimiter="/")

    # We need to consume the iterator to populate prefixes
    _ = list(blobs)

    folders = []
    for p in blobs.prefixes:
        # p looks like "raw-manga/folder_name/"
        folder_name = p[len(prefix) :].rstrip("/")
        if folder_name:
            folders.append(folder_name)

    return sorted(folders)


def folder_exists(folder_name: str) -> bool:
    """Check if a manga folder exists under raw-manga/."""
    client = get_gcs_client()
    bucket_name = _get_bucket_name()
    bucket = client.bucket(bucket_name)

    prefix = f"{_RAW_PREFIX}/{folder_name}/"
    blobs = client.list_blobs(bucket, prefix=prefix, max_results=1)
    return any(True for _ in blobs)


def list_images_in_folder(folder_name: str) -> list[str]:
    """List image filenames in a raw-manga/<folder_name>/ directory."""
    client = get_gcs_client()
    bucket_name = _get_bucket_name()
    bucket = client.bucket(bucket_name)

    prefix = f"{_RAW_PREFIX}/{folder_name}/"
    blobs = client.list_blobs(bucket, prefix=prefix)

    images = []
    for blob in blobs:
        name = blob.name[len(prefix) :]
        # Skip subdirectories or empty names
        if not name or "/" in name:
            continue
        ext = Path(name).suffix.lower()
        if ext in _IMAGE_EXTENSIONS:
            images.append(name)

    return sorted(images)


def download_images(
    folder_name: str,
    filenames: list[str],
    local_dir: Path,
) -> list[Path]:
    """
    Download images from raw-manga/<folder_name>/ to a local directory.

    Returns list of local file paths for successfully downloaded images.
    """
    bucket = get_bucket()
    download_dir = local_dir / "raw" / folder_name
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for filename in filenames:
        blob_path = f"{_RAW_PREFIX}/{folder_name}/{filename}"
        blob = bucket.blob(blob_path)
        local_path = download_dir / filename
        try:
            blob.download_to_filename(str(local_path))
            downloaded.append(local_path)
            logger.info("Downloaded: %s", blob_path)
        except Exception as e:
            logger.error("Failed to download %s: %s", blob_path, e)

    return downloaded


def upload_translated_image(
    folder_name: str,
    filename: str,
    local_path: Path,
) -> None:
    """Upload a translated image to translated-manga/<folder_name>/."""
    bucket = get_bucket()
    blob_path = f"{_TRANSLATED_PREFIX}/{folder_name}/{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    logger.info("Uploaded: %s", blob_path)


def upload_translated_images(
    folder_name: str,
    translated_paths: list[Path],
) -> None:
    """Upload all translated images to the bucket."""
    for local_path in translated_paths:
        upload_translated_image(folder_name, local_path.name, local_path)


def generate_download_url(folder_name: str, ttl_hours: int = 24) -> str:
    """
    Generate a presigned download URL for translated manga.

    Zips all files in translated-manga/<folder_name>/, uploads the zip,
    and returns a signed URL valid for `ttl_hours`.
    """
    client = get_gcs_client()
    bucket_name = _get_bucket_name()
    bucket = client.bucket(bucket_name)

    prefix = f"{_TRANSLATED_PREFIX}/{folder_name}/"
    blobs = list(client.list_blobs(bucket, prefix=prefix))

    if not blobs:
        raise ValueError(f"No translated images found for '{folder_name}'")

    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for blob in blobs:
            filename = blob.name[len(prefix) :]
            if not filename or "/" in filename:
                continue
            data = blob.download_as_bytes()
            zf.writestr(filename, data)

    zip_buffer.seek(0)

    # Upload zip as Chapter.cbz (CBZ is the popular manga archive format)
    zip_blob_path = f"{_TRANSLATED_PREFIX}/{folder_name}/Chapter.cbz"
    zip_blob = bucket.blob(zip_blob_path)
    zip_blob.upload_from_file(zip_buffer, content_type="application/zip")
    logger.info("Uploaded cbz: %s", zip_blob_path)

    # Generate signed URL
    url = zip_blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=ttl_hours),
        method="GET",
    )

    return url


def cleanup_local_files(local_dir: Path, folder_name: str) -> None:
    """Remove downloaded raw and translated local files after upload."""
    raw_dir = local_dir / "raw" / folder_name
    translated_dir = local_dir / "translated" / folder_name
    for d in [raw_dir, translated_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            logger.info("Cleaned up local directory: %s", d)

    # Also clean up empty parent dirs
    for parent in [local_dir / "raw", local_dir / "translated"]:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

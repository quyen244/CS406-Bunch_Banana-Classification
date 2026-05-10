"""
conftest.py — Shared fixtures cho pytest test suite

Cung cấp:
  - base_url: URL của Gateway
  - sample_image_bytes: ảnh PNG giả hợp lệ (3x3 pixels xanh lá)
  - sample_text_bytes: bytes không phải ảnh (để test validation)
"""

import io
import struct
import zlib

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GATEWAY_BASE_URL = "http://localhost:8080"


# ---------------------------------------------------------------------------
# Helpers: tạo ảnh PNG tối thiểu mà không cần Pillow/numpy
# ---------------------------------------------------------------------------

def _make_minimal_png(width: int = 4, height: int = 4) -> bytes:
    """
    Tạo một file PNG hợp lệ tối thiểu (solid green) mà không dùng thư viện ảnh.
    PNG spec: signature + IHDR + IDAT + IEND
    """
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # IHDR: width, height, bit_depth=8, color_type=2 (RGB), compression=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT: raw pixel data (filter byte 0x00 per row + RGB pixels)
    raw_rows = b""
    for _ in range(height):
        # Filter byte + width * 3 bytes RGB (green: 0x00, 0xFF, 0x00)
        row = b"\x00" + b"\x00\xFF\x00" * width
        raw_rows += row
    compressed = zlib.compress(raw_rows)
    idat = _chunk(b"IDAT", compressed)

    iend = _chunk(b"IEND", b"")

    png_signature = b"\x89PNG\r\n\x1a\n"
    return png_signature + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url() -> str:
    """URL base của Gateway (chạy local hoặc trong Docker)."""
    return GATEWAY_BASE_URL


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """
    Ảnh PNG hợp lệ tối thiểu (4x4 pixels, solid green).
    Dùng làm input hợp lệ cho các test predict.
    """
    return _make_minimal_png(width=4, height=4)


@pytest.fixture(scope="session")
def sample_text_bytes() -> bytes:
    """Bytes không phải ảnh — dùng để test validation từ chối file sai type."""
    return b"This is not an image file. Plain text content."

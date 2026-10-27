import io
import socket

import pytest
from PIL import Image, UnidentifiedImageError

from engine.media import PublicResolver, normalize, validate_url
from engine.workflow import build


@pytest.mark.parametrize(
    "url",
    [
        "http://①②⑦.⓪.⓪.①/",
        "http://2130706433/",
        "http://127.1/",
        "http://[64:ff9b::a00:1]/",
        "http://localhost/",
        "http://224.0.0.1/a",
        "http://127.0.0.1/a",
        "http://10.0.0.1/",
        "http://[::1]/",
        "http://169.254.169.254/",
        "file:///etc/passwd",
        "https://user:password@example.org/x",
        "http://example.org:8188/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
def test_private_or_unsafe_urls(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_image_decode_and_metadata_removal():
    source = io.BytesIO()
    exif = Image.Exif()
    exif[270] = "private metadata"
    Image.new("RGB", (32, 24), "red").save(source, "JPEG", exif=exif)
    data = normalize(source.getvalue())
    assert Image.open(io.BytesIO(data)).size == (32, 24)
    assert b"private metadata" not in data
    with pytest.raises(UnidentifiedImageError):
        normalize(b"<html>not an image</html>")


async def test_dns_resolution_blocks_internal(monkeypatch):
    import asyncio

    async def lookup(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)
    with pytest.raises(ValueError):
        await PublicResolver().resolve("example.org", 80)


def test_generation_and_multiple_reference_graph():
    args = ("draw fox", 42, 1024, 768)
    graph = build(*args, [], "model", "encoder", "vae")
    assert graph["4"]["inputs"]["text"] == "draw fox"
    assert graph["8"]["inputs"]["height"] == graph["9"]["inputs"]["height"] == 768
    edit = build(*args, ["a.png", "b.png"], "model", "encoder", "vae")
    assert edit["6"]["inputs"]["conditioning"] == ["27", 0]
    assert edit["27"]["inputs"]["conditioning"] == ["23", 0]
    assert edit["20"]["inputs"]["image"] == "a.png"
    assert edit["24"]["inputs"]["image"] == "b.png"
    assert graph["6"]["inputs"]["conditioning"] == ["4", 0]
    with pytest.raises(ValueError):
        build("", 1, 1024, 1024, [], "m", "e", "v")


async def test_public_dns_resolution_is_usable(monkeypatch):
    import asyncio

    async def lookup(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)
    records = await PublicResolver().resolve("example.org", 443)
    assert records[0]["host"] == "8.8.8.8"
    validate_url("https://example.org/reference.png")
    validate_url("https://8.8.8.8/reference.png")

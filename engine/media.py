"""Bounded, decoded images and public-network-only URL fetching."""

import asyncio
import io
import ipaddress
import socket
import warnings
from urllib.parse import urlsplit

import aiohttp
from PIL import Image, ImageOps

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000


def public_address(ip):
    mapped = getattr(ip, "ipv4_mapped", None)
    return (
        ip.is_global
        and not ip.is_multicast
        and not ip.is_reserved
        and (mapped is None or public_address(mapped))
    )


class PublicResolver(aiohttp.abc.AbstractResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        records = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM, family=family
        )
        result = []
        for fam, _, proto, _, addr in records:
            ip = ipaddress.ip_address(addr[0])
            if not public_address(ip):
                raise ValueError("Reference URLs must resolve to public Internet addresses.")
            result.append(
                dict(
                    hostname=host,
                    host=addr[0],
                    port=port,
                    family=fam,
                    proto=proto,
                    flags=socket.AI_NUMERICHOST,
                )
            )
        return result

    async def close(self):
        pass


def validate_url(url: str):
    parts = urlsplit(url)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("Use an HTTP(S) reference image URL without credentials.")
    if parts.port not in (None, 80, 443):
        raise ValueError("Reference URLs must use port 80 or 443.")
    # aiohttp bypasses resolvers for literal IPs.
    try:
        ip = ipaddress.ip_address(parts.hostname)
    except ValueError:
        return
    if not public_address(ip):
        raise ValueError("Private-network reference URLs are not allowed.")


async def read_bounded(response, limit=MAX_BYTES):
    response.raise_for_status()
    data = bytearray()
    async for chunk in response.content.iter_chunked(65536):
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError("Image exceeds the 20 MiB limit.")
    return bytes(data)


async def download(url: str) -> bytes:
    # Resolve and connect using the same validated address list; no DNS preflight race.
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(resolver=PublicResolver()),
        timeout=aiohttp.ClientTimeout(total=40),
        trust_env=False,
    ) as session:
        for _ in range(4):
            validate_url(url)
            async with session.get(url, allow_redirects=False) as response:
                if response.status in (301, 302, 303, 307, 308):
                    from urllib.parse import urljoin

                    url = urljoin(url, response.headers.get("Location", ""))
                    continue
                return await read_bounded(response)
    raise ValueError("Too many image redirects.")


def normalize(data: bytes) -> bytes:
    if len(data) > MAX_BYTES:
        raise ValueError("Image exceeds the 20 MiB limit.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 25 megapixel limit.")
            image.load()
            result = io.BytesIO()
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            clean = Image.new("RGB", oriented.size)
            clean.paste(oriented)
            clean.save(result, "PNG")
            output = result.getvalue()
            if len(output) > MAX_BYTES:
                raise ValueError("Decoded image exceeds the 20 MiB limit.")
            return output

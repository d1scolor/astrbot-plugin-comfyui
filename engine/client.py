"""Only the administrator-configured ComfyUI endpoint is trusted for LAN access."""

import json
from urllib.parse import urlsplit

import aiohttp

from .media import normalize, read_bounded


class ComfyClient:
    def __init__(self, url: str, token: str = ""):
        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username
            or parts.password
        ):
            raise ValueError("Configure an HTTP(S) ComfyUI endpoint without embedded credentials.")
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers, trust_env=False, timeout=aiohttp.ClientTimeout(total=40)
        )

    async def close(self):
        if self.session:
            await self.session.close()

    async def request(self, method, path, **kwargs):
        async with self.session.request(
            method, self.url + path, allow_redirects=False, **kwargs
        ) as r:
            if r.status >= 300:
                # Do not expose backend bodies (which can contain paths/credentials) to the model.
                raise RuntimeError(f"ComfyUI HTTP {r.status}")
            return json.loads(await read_bounded(r, 8 * 1024 * 1024))

    async def upload(self, data: bytes, name: str):
        form = aiohttp.FormData()
        form.add_field("image", data, filename=name, content_type="image/png")
        form.add_field("type", "input")
        form.add_field("overwrite", "false")
        result = await self.request("POST", "/upload/image", data=form)
        return (result.get("subfolder", "").strip("/") + "/" + result["name"]).lstrip("/")

    async def submit(self, graph, job_id):
        result = await self.request(
            "POST",
            "/prompt",
            json={"prompt": graph, "client_id": job_id, "extra_data": {"direct_job_id": job_id}},
        )
        if "prompt_id" not in result or result.get("node_errors"):
            raise RuntimeError("ComfyUI rejected the workflow.")
        return result["prompt_id"]

    async def history(self, prompt_id):
        result = await self.request("GET", f"/history/{prompt_id}")
        return result.get(prompt_id)

    async def image(self, descriptor):
        params = {key: descriptor.get(key, "") for key in ("filename", "subfolder", "type")}
        if params["type"] != "output":
            raise ValueError("Expected a saved output image.")
        async with self.session.get(self.url + "/view", params=params, allow_redirects=False) as r:
            if r.status >= 300:
                raise RuntimeError("ComfyUI output download failed.")
            return normalize(await read_bounded(r))

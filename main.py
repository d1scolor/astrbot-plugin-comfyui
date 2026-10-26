"""AstrBot adapter. Media stays inside the plugin; the model never sends local paths."""

import asyncio
import base64
import hashlib
import json
import secrets
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.star import Context, Star, StarTools, register

from .engine.client import ComfyClient
from .engine.jobs import Jobs
from .engine.media import MAX_BYTES, download, normalize
from .engine.workflow import build


@register("astrbot_plugin_comfyui_direct", "d1scolor", "Direct ComfyUI image jobs", "0.1.1")
class ComfyDirect(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.client = None
        self.jobs = None
        self.init_lock = asyncio.Lock()

    async def initialize(self):
        async with self.init_lock:
            if self.jobs:
                return
            client = ComfyClient(self.config.get("comfy_url", ""), self.config.get("api_token", ""))
            await client.start()
            try:
                jobs = Jobs(
                    StarTools.get_data_dir("astrbot_plugin_comfyui_direct"),
                    client,
                    timeout=max(30, min(int(self.config.get("job_timeout", 600)), 3600)),
                    queue_limit=max(1, min(int(self.config.get("queue_limit", 8)), 32)),
                )
                await jobs.recover()
            except Exception:
                await client.close()
                raise
            self.client, self.jobs = client, jobs

    async def terminate(self):
        if self.jobs:
            await self.jobs.close()
        if self.client:
            await self.client.close()

    @staticmethod
    def identity(event):
        return event.unified_msg_origin, event.get_sender_id(), event.message_obj.message_id

    @staticmethod
    def sender(event):
        # Capture THIS event, not the last message received in a shared group session.
        async def send(paths, text):
            chain = [Plain(text)] + [Image.fromFileSystem(str(path)) for path in paths]
            await event.send(MessageChain(chain=chain))

        return send

    @staticmethod
    def attachments(event):
        components = event.get_messages()
        images = [c for c in components if isinstance(c, Image)]
        for component in components:
            if isinstance(component, Reply):
                images.extend(c for c in component.chain or [] if isinstance(c, Image))
        return images

    async def attachment_bytes(self, image):
        ref = image.url or image.file or ""
        if ref.startswith(("http://", "https://")):
            return await download(ref)
        if ref.startswith("base64://"):
            if len(ref) > MAX_BYTES * 4 // 3 + 32:
                raise ValueError("Attachment is too large.")
            return base64.b64decode(ref[9:], validate=True)
        # Only platform-provided Image components reach this branch. Tool args cannot name files.
        from urllib.parse import urlsplit

        if urlsplit(ref).scheme not in ("", "file"):
            raise ValueError("Unsupported platform attachment format.")
        path = Path(await image.convert_to_file_path())
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError("Attachment is unavailable or too large.")
        return path.read_bytes()

    async def submit(self, event, prompt, reference_urls, use_attached_images, edit):
        try:
            await self.initialize()
            if not isinstance(reference_urls, list) or any(
                not isinstance(u, str) for u in reference_urls
            ):
                raise ValueError("reference_urls must be a list of public image URLs.")
            attached = self.attachments(event) if use_attached_images else []
            if len(attached) + len(reference_urls) > 3:
                raise ValueError("Select at most three images; quote only the intended references.")
            if edit and not attached and not reference_urls:
                raise ValueError("Editing requires an attached, quoted or public-URL image.")
            # Validate prompt before network work.
            seed = secrets.randbits(63)
            width, height = (
                int(self.config.get("width", 1024)),
                int(self.config.get("height", 1024)),
            )
            models = [
                self.config.get(k, default)
                for k, default in (
                    ("model", "flux-2-klein-4b.safetensors"),
                    ("encoder", "qwen_3_4b.safetensors"),
                    ("vae", "flux2-vae.safetensors"),
                )
            ]
            build(prompt, seed, width, height, [], *models)
            data = [await self.attachment_bytes(image) for image in attached]
            data.extend([await download(url) for url in reference_urls])
            data = [await asyncio.to_thread(normalize, image) for image in data]
            signature = [prompt, edit, [hashlib.sha256(image).hexdigest() for image in data]]
            uploads = [
                await self.client.upload(image, f"direct-{secrets.token_hex(12)}.png")
                for image in data
            ]
            graph = build(prompt, seed, width, height, uploads, *models)
            origin, owner, message_id = self.identity(event)
            row = await self.jobs.enqueue(
                origin, owner, message_id, signature, graph, self.sender(event)
            )
            return json.dumps(
                {
                    "job_id": row["id"],
                    "state": row["state"],
                    "reference_count": len(uploads),
                    "delivery": "Plugin sends image automatically. Do not send paths, "
                    "call send_message_to_user, wait tools or claim delivery before state=delivered.",
                }
            )
        except ValueError as error:
            return json.dumps({"error": str(error)})
        except Exception:
            return json.dumps(
                {
                    "error": "Image request failed. Check plugin endpoint, connectivity and configuration."
                }
            )

    @filter.llm_tool(name="comfy_generate_image")
    async def generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        reference_urls: list[str] | None = None,
        use_attached_images: bool = False,
    ) -> str:
        """Generate an image, optionally conditioned on references. Plugin sends the result automatically.
        Research/search belongs to other tools; a description alone is not a visual reference.

        Args:
            prompt(string): Full visual prompt, including researched character details if needed.
            reference_urls(array[string]): Public HTTP(S) image URLs obtained from other tools, up to three.
            use_attached_images(boolean): Use images attached to or quoted in this request as references.
        """
        return await self.submit(event, prompt, reference_urls or [], use_attached_images, False)

    @filter.llm_tool(name="comfy_edit_image")
    async def edit(
        self,
        event: AstrMessageEvent,
        instruction: str,
        reference_urls: list[str] | None = None,
        use_attached_images: bool = True,
    ) -> str:
        """Edit attached/quoted images or public image URLs. Sends results directly; never invent image inputs.

        Args:
            instruction(string): Requested edits. Refer to multiple inputs as image 1, image 2, image 3.
            reference_urls(array[string]): Additional public HTTP(S) image URLs, up to three total inputs.
            use_attached_images(boolean): Include current-message and quoted images, in that order.
        """
        return await self.submit(
            event, instruction, reference_urls or [], use_attached_images, True
        )

    @filter.llm_tool(name="comfy_image_status")
    async def status(
        self, event: AstrMessageEvent, job_id: str = "", retry_delivery: bool = False
    ) -> str:
        """Check your own job in this conversation. Recover completed images after a restart.

        Args:
            job_id(string): Job ID returned by generation/edit, or empty to list your recent jobs.
            retry_delivery(boolean): Only true if user explicitly requested delivery retry; uncertain sends can duplicate.
        """
        try:
            await self.initialize()
            origin, owner, _ = self.identity(event)
            if not job_id:
                rows = self.jobs.db.execute(
                    "SELECT id, state, error FROM jobs WHERE origin=? AND owner=? ORDER BY created DESC LIMIT 5",
                    (origin, owner),
                ).fetchall()
                return json.dumps({"jobs": [dict(row) for row in rows]})
            row = self.jobs.owned(job_id, origin, owner)
            if row["state"] == "generated" or retry_delivery:
                await self.jobs.deliver(job_id, self.sender(event), allow_retry=retry_delivery)
            row = self.jobs.owned(job_id, origin, owner)
            return json.dumps({"job_id": job_id, "state": row["state"], "detail": row["error"]})
        except ValueError:
            return json.dumps(
                {"error": "Job not found in your conversation or configuration invalid."}
            )

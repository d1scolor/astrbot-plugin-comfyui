"""Adapter contract tests with event/message doubles; no QQ messages or SDK runtime required."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_jobs import Backend, settle

from engine.jobs import Jobs


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    class Component:
        def __init__(self, value=None, chain=None):
            self.value, self.chain = value, chain

    class Image(Component):
        @staticmethod
        def fromFileSystem(path):
            return Image(path)

    class Star:
        def __init__(self, context):
            self.context = context

    def decorator(*args, **kwargs):
        return lambda fn: fn

    modules = {
        "astrbot": {},
        "astrbot.api": {"AstrBotConfig": dict},
        "astrbot.api.event": {
            "AstrMessageEvent": object,
            "MessageChain": Component,
            "filter": SimpleNamespace(
                llm_tool=decorator, command=decorator, on_llm_request=decorator
            ),
        },
        "astrbot.api.message_components": {"Image": Image, "Plain": Component, "Reply": Component},
        "astrbot.api.star": {
            "Context": object,
            "Star": Star,
            "StarTools": object,
            "register": decorator,
        },
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parent.parent
    package = types.ModuleType("test_plugin")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "test_plugin", package)
    spec = importlib.util.spec_from_file_location("test_plugin.main", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin = module.ComfyDirect(None, {})
    plugin.jobs = Jobs(tmp_path, Backend())
    plugin.client = plugin.jobs.client
    return plugin, module


class Event:
    def __init__(self, origin="group-a", owner="alice", message="m"):
        self.unified_msg_origin = origin
        self.owner = owner
        self.message_obj = SimpleNamespace(message_id=message)
        self.sent = []

    def get_sender_id(self):
        return self.owner

    def get_messages(self):
        return []

    async def send(self, message):
        self.sent.append(message)


async def test_tool_owns_delivery_and_never_returns_path(adapter):
    import json

    plugin, _ = adapter
    event = Event()
    result = json.loads(await plugin.generate(event, "a fox"))
    assert result["state"] == "queued"
    assert "local_path" not in result and "/" not in result["job_id"]
    await settle(plugin.jobs)
    assert len(event.sent) == 1
    assert len(event.sent[0].chain) == 1
    assert isinstance(event.sent[0].chain[0], _.Image)
    foreign = Event(owner="bob")
    denied = json.loads(await plugin.status(foreign, result["job_id"]))
    assert "error" in denied
    await plugin.jobs.close()


async def test_edit_never_fakes_missing_reference(adapter):
    import json

    plugin, _ = adapter
    result = json.loads(await plugin.edit(Event(), "make it blue"))
    assert "requires" in result["error"]
    assert plugin.client.submissions == 0
    await plugin.jobs.close()


async def test_current_and_quoted_reference_images(adapter):
    plugin, module = adapter
    first, quoted = module.Image("first"), module.Image("quoted")
    event = Event()
    event.get_messages = lambda: [first, module.Reply(chain=[quoted])]
    assert plugin.attachments(event) == [first, quoted]
    await plugin.jobs.close()


async def test_edit_uploads_and_conditions_on_supplied_image(adapter):
    import io
    import json

    from PIL import Image as PILImage

    plugin, module = adapter
    image = io.BytesIO()
    PILImage.new("RGB", (16, 16), "blue").save(image, "PNG")
    event = Event()
    event.get_messages = lambda: [module.Image("platform-supplied")]
    seen = {}

    async def attachment_bytes(component):
        return image.getvalue()

    async def upload(data, name):
        seen["upload"] = data
        return "reference.png"

    original = plugin.client.submit

    async def submit(graph, job):
        seen["graph"] = graph
        return await original(graph, job)

    plugin.attachment_bytes = attachment_bytes
    plugin.client.upload = upload
    plugin.client.submit = submit
    result = json.loads(await plugin.edit(event, "make it red"))
    await settle(plugin.jobs)
    assert result["reference_count"] == 1
    assert seen["graph"]["20"]["inputs"]["image"] == "reference.png"
    assert seen["graph"]["6"]["inputs"]["conditioning"] == ["23", 0]
    assert len(event.sent) == 1
    await plugin.jobs.close()


@pytest.mark.parametrize("endpoint", ["", "not-a-url", "ftp://example.org"])
async def test_unconfigured_plugin_loads_and_tools_explain_settings(adapter, endpoint):
    import json

    plugin, _ = adapter
    await plugin.jobs.close()
    plugin.jobs = None
    plugin.client = None
    plugin.config = {"comfy_url": endpoint}
    await plugin.initialize()
    assert plugin.jobs is None
    result = json.loads(await plugin.generate(Event(), "a fox"))
    assert "Configure comfy_url" in result["error"]
    result = json.loads(await plugin.status(Event()))
    assert "Configure comfy_url" in result["error"]
    await plugin.terminate()


async def test_generate_then_edit_previous_image_and_repeated_tool_call(adapter):
    import json

    plugin, _ = adapter

    async def upload(data, name):
        return "reused.png"

    plugin.client.upload = upload
    generated = json.loads(await plugin.generate(Event(), "fox in a library"))
    await settle(plugin.jobs)
    image_id = plugin.jobs.output_images(plugin.jobs.get(generated["job_id"]))[0]["id"]
    followup = Event(message="next-turn")
    edited = json.loads(await plugin.edit(followup, "change background to snow"))
    assert edited["input_image_ids"] == [image_id]
    await settle(plugin.jobs)
    repeated = json.loads(await plugin.edit(followup, "change background to snow"))
    assert repeated["job_id"] == edited["job_id"]
    assert repeated["input_image_ids"] == [image_id]
    assert plugin.client.submissions == 2
    await plugin.jobs.close()


async def test_explicit_id_and_model_context_are_user_scoped(adapter):
    import json

    plugin, _ = adapter
    image = plugin.jobs.images.add(
        "group-a", "alice", "url", "key", b"validated", "Character reference"
    )

    async def upload(data, name):
        return "reused.png"

    plugin.client.upload = upload
    own = Event()
    result = json.loads(await plugin.edit(own, "red scarf", image_ids=[image["id"]]))
    assert result["input_image_ids"] == [image["id"]]
    await settle(plugin.jobs)
    denied = json.loads(await plugin.edit(Event(owner="bob"), "red scarf", image_ids=[image["id"]]))
    assert "error" in denied
    denied = json.loads(
        await plugin.edit(Event(origin="group-b"), "red scarf", image_ids=[image["id"]])
    )
    assert "error" in denied
    request = SimpleNamespace(system_prompt="Persona")
    await plugin.image_context(own, request)
    assert request.system_prompt.startswith("Persona")
    assert image["id"] in request.system_prompt
    foreign_request = SimpleNamespace(system_prompt="Persona")
    await plugin.image_context(Event(owner="bob"), foreign_request)
    assert image["id"] not in foreign_request.system_prompt
    listing = json.loads(await plugin.list_images(own))
    assert any(row["id"] == image["id"] for row in listing["images"])
    assert all("filename" not in row and "source_key" not in row for row in listing["images"])
    await plugin.jobs.close()


async def test_incoming_image_is_registered_without_generation(adapter):
    import io
    import json

    from PIL import Image as PILImage

    plugin, module = adapter
    buffer = io.BytesIO()
    PILImage.new("RGB", (16, 16)).save(buffer, "PNG")
    event = Event()
    event.get_messages = lambda: [module.Image("incoming")]
    reads = []

    async def attachment_bytes(component):
        reads.append(True)
        return buffer.getvalue()

    plugin.attachment_bytes = attachment_bytes
    request = SimpleNamespace(system_prompt="")
    await plugin.image_context(event, request)
    listed = json.loads(await plugin.list_images(event))
    assert len(listed["images"]) == 1
    assert len(reads) == 1
    assert listed["images"][0]["id"] in request.system_prompt
    assert plugin.client.submissions == 0
    await plugin.jobs.close()


async def test_ambiguous_latest_output_requires_explicit_selection(adapter):
    import json

    plugin, _ = adapter
    for index in range(2):
        plugin.jobs.images.add(
            "group-a", "alice", "generated", f"samejob:{index}", b"validated", "Output"
        )
    result = json.loads(await plugin.edit(Event(), "make it blue"))
    assert "multiple images" in result["error"]
    assert plugin.client.submissions == 0
    await plugin.jobs.close()


async def test_search_reference_import_and_reuse_without_redownload(adapter, monkeypatch):
    import io
    import json

    from PIL import Image as PILImage

    plugin, module = adapter
    buffer = io.BytesIO()
    PILImage.new("RGB", (16, 16), "red").save(buffer, "PNG")
    downloads = []

    async def download(url):
        downloads.append(url)
        return buffer.getvalue()

    async def upload(data, name):
        return "reference.png"

    monkeypatch.setattr(module, "download", download)
    plugin.client.upload = upload
    result = json.loads(
        await plugin.import_image(
            Event(), "https://example.org/character.png", "Character reference"
        )
    )
    image_id = result["image"]["id"]
    assert "url" not in result["image"] and "filename" not in result["image"]
    edited = json.loads(
        await plugin.edit(Event(message="next"), "winter coat", image_ids=[image_id])
    )
    await settle(plugin.jobs)
    assert edited["input_image_ids"] == [image_id]
    assert len(downloads) == 1
    await plugin.jobs.close()


async def test_reusable_generated_handle_survives_restart(adapter):
    import json

    plugin, _ = adapter
    backend = plugin.client
    result = json.loads(await plugin.generate(Event(), "fox"))
    await settle(plugin.jobs)
    before = plugin.jobs.output_images(plugin.jobs.get(result["job_id"]))
    directory = plugin.jobs.directory
    await plugin.jobs.close()
    plugin.jobs = Jobs(directory, backend)
    await plugin.jobs.recover()
    after = plugin.jobs.output_images(plugin.jobs.get(result["job_id"]))
    assert before == after

    async def upload(data, name):
        return "persisted.png"

    backend.upload = upload
    edited = json.loads(await plugin.edit(Event(message="after-restart"), "snowy background"))
    assert edited["input_image_ids"] == [before[0]["id"]]
    await settle(plugin.jobs)
    await plugin.jobs.close()


async def test_edit_failure_cannot_silently_fall_back_to_text_generation(adapter):
    import json

    plugin, _ = adapter
    event = Event()
    failed = json.loads(await plugin.edit(event, "change its background"))
    assert "error" in failed
    fallback = json.loads(await plugin.generate(event, "a new similar image"))
    assert "text-only fallback is blocked" in fallback["error"]
    assert plugin.client.submissions == 0
    # A new user request may intentionally ask for an unrelated text generation.
    fresh = json.loads(await plugin.generate(Event(message="new-request"), "a new picture"))
    assert fresh["state"] == "queued"
    await settle(plugin.jobs)
    await plugin.jobs.close()

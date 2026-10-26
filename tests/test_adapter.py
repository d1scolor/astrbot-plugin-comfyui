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
            "filter": SimpleNamespace(llm_tool=decorator, command=decorator),
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
    assert len(event.sent[0].chain) == 2
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

import io

import pytest
from PIL import Image

from engine.postprocess import append, existing, glyphs, plan, text_overlay
from engine.workflow import build


def test_chinese_caption_has_real_glyphs_alpha_outline_and_rotation():
    operation = plan([{"operation": "text", "text": "哭哭哦", "rotation": -20}], 700, 500)[0][0]
    assert all(ord(c) in glyphs() for c in "哭哭哦")
    data = text_overlay(700, 500, operation)
    overlay = Image.open(io.BytesIO(data))
    assert overlay.mode == "RGBA" and overlay.size == (700, 500)
    assert overlay.getpixel((0, 0))[3] == 0
    assert overlay.getbbox() is not None
    rgb = overlay.convert("RGB")
    colors = {rgb.getpixel((x, y)) for x in range(rgb.width) for y in range(rgb.height)}
    assert (255, 255, 255) in colors and (0, 0, 0) in colors
    background = Image.new("RGBA", overlay.size, (10, 20, 30, 255))
    result = Image.alpha_composite(background, overlay)
    assert result.getpixel((0, 0)) == background.getpixel((0, 0))


@pytest.mark.parametrize(
    "operations",
    [
        [{"operation": "exec", "command": "anything"}],
        [{"operation": "text", "text": "hi", "font_path": "/arbitrary/path"}],
        [{"operation": "text", "text": "hi", "rotation": float("nan")}],
        [{"operation": "text", "text": "x" * 241}],
        [{"operation": "crop", "width": 9999, "height": 1}],
        [{"operation": "resize", "width": 4096, "height": 4096}],
        [{"operation": "resize", "width": True, "height": 512}],
        [{"operation": "text", "text": "hi", "style": "unknown"}],
    ],
)
def test_invalid_operations_are_rejected(operations):
    with pytest.raises(ValueError):
        plan(operations, 512, 512)


async def test_generation_and_postprocess_share_one_graph_and_one_save():
    graph = build("fox", 1, 512, 512, [], "model", "encoder", "vae")
    uploads = []

    async def upload(data, name):
        uploads.append(data)
        return "caption.png"

    await append(
        graph,
        ["11", 0],
        512,
        512,
        [
            {"operation": "resize", "width": 640, "height": 480},
            {"operation": "text", "text": "哭哭哦", "rotation": -20},
            {"operation": "crop", "x": 0, "y": 0, "width": 600, "height": 400},
        ],
        upload,
    )
    assert sum(node["class_type"] == "SaveImage" for node in graph.values()) == 1
    assert graph["12"]["inputs"]["images"] == ["1004", 0]
    assert graph["1001"]["inputs"]["image"] == "caption.png"
    assert graph["1002"]["class_type"] == "InvertMask"
    assert graph["1003"]["inputs"]["destination"] == ["1000", 0]
    assert Image.open(io.BytesIO(uploads[0])).size == (640, 480)


async def test_existing_processing_has_no_model_nodes():
    graph = existing("input.png")

    async def upload(data, name):
        return "caption.png"

    await append(graph, ["1", 0], 512, 512, [{"operation": "text", "text": "哭哭哦"}], upload)
    assert not any(
        "Loader" in node["class_type"] or "Sampler" in node["class_type"] for node in graph.values()
    )


def test_long_caption_shrinks_without_huge_allocation():
    op = plan(
        [{"operation": "text", "text": "哭" * 200, "font_size": 0.4, "rotation": 0}], 512, 512
    )[0][0]
    result = Image.open(io.BytesIO(text_overlay(512, 512, op)))
    assert result.size == (512, 512)


def test_default_caption_is_stylized_but_plain_style_stays_upright():
    meme = plan([{"operation": "text", "text": "哭哭哦"}], 512, 512)[0][0]
    assert meme["style"] == "meme" and meme["rotation"] == -10
    plain = plan([{"operation": "text", "text": "字", "style": "plain_white"}], 512, 512)[0][0]
    assert plain["rotation"] == 0

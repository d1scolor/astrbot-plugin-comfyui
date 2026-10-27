"""Validated image operations, compiled into existing ComfyUI core nodes."""

import asyncio
import io
import math
import secrets
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).resolve().parent.parent / "assets/fonts/NotoSansCJKsc-Bold.otf"
STYLES = {
    "meme": ("white", "black"),
    "white_black_outline": ("white", "black"),
    "black_white_outline": ("black", "white"),
    "plain_white": ("white", None),
    "plain_black": ("black", None),
}
POSITIONS = {f"{y}_{x}" for y in ("top", "center", "bottom") for x in ("left", "center", "right")}


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}.")
    return value


def number(value, low, high, name):
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a number between {low} and {high}.")
    return value


def plan(operations, width, height):
    if not isinstance(operations, list) or len(operations) > 6:
        raise ValueError("postprocess must be a list of at most six operations.")
    result = []
    for item in operations:
        if not isinstance(item, dict):
            raise ValueError("Each postprocess operation must be an object.")
        op = item.get("operation")
        if op == "crop":
            allowed = {"operation", "x", "y", "width", "height"}
            x = integer(item.get("x", 0), 0, width - 1, "crop x")
            y = integer(item.get("y", 0), 0, height - 1, "crop y")
            w = integer(item.get("width"), 1, width - x, "crop width")
            h = integer(item.get("height"), 1, height - y, "crop height")
            clean = {"operation": op, "x": x, "y": y, "width": w, "height": h}
            width, height = w, h
        elif op == "resize":
            allowed = {"operation", "width", "height"}
            width = integer(item.get("width"), 16, 4096, "resize width")
            height = integer(item.get("height"), 16, 4096, "resize height")
            clean = {"operation": op, "width": width, "height": height}
        elif op == "text":
            allowed = {"operation", "text", "position", "font_size", "style", "rotation"}
            text = item.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 240:
                raise ValueError("Caption text must contain 1–240 characters.")
            if text.count("\n") > 5 or any(ord(c) < 32 and c != "\n" for c in text):
                raise ValueError("Caption supports up to six lines and no control characters.")
            position = item.get("position", "top_right")
            if position == "center":
                position = "center_center"
            if not isinstance(position, str) or position not in POSITIONS:
                raise ValueError(
                    "Caption position must be top/center/bottom_left/center/right, or center."
                )
            style = item.get("style", "meme")
            if not isinstance(style, str) or style not in STYLES:
                raise ValueError("Unknown caption style.")
            clean = {
                "operation": op,
                "text": text,
                "position": position,
                "style": style,
                "font_size": number(item.get("font_size", 0.12), 0.02, 0.4, "font_size"),
                "rotation": number(
                    item.get("rotation", -10 if style == "meme" else 0), -180, 180, "rotation"
                ),
            }
        else:
            raise ValueError("Supported postprocess operations: text, crop, resize.")
        if set(item) - allowed:
            raise ValueError(
                "Unknown postprocess fields. Arbitrary nodes, paths and scripts are not accepted."
            )
        if width > 4096 or height > 4096 or width * height > 8_000_000:
            raise ValueError(
                "Postprocess canvas exceeds 4096 pixels per side or 8 megapixels. Resize/crop first."
            )
        result.append((clean, width, height))
    return result


@lru_cache(maxsize=1)
def glyphs():
    if not FONT.is_file():
        raise ValueError("Bundled Chinese caption font is missing. Reinstall/update the plugin.")
    with TTFont(FONT) as font:
        return frozenset(font.getBestCmap())


def text_overlay(width, height, operation):
    text = operation["text"]
    if any(not c.isspace() and ord(c) not in glyphs() for c in text):
        raise ValueError("Caption contains characters not supported by the bundled CJK font.")
    fill, outline = STYLES[operation["style"]]
    margin = max(2, round(min(width, height) * 0.04))
    max_width, max_height = width - 2 * margin, height - 2 * margin
    if min(max_width, max_height) < 10:
        raise ValueError("Image is too small for a readable caption.")
    size = min(1024, max(10, round(height * operation["font_size"])))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    while size >= 10:
        font = ImageFont.truetype(str(FONT), size)
        stroke = (
            max(1, round(size * (0.12 if operation["style"] == "meme" else 0.08))) if outline else 0
        )
        lines = []
        for raw in text.split("\n"):
            line = ""
            for char in raw:
                if (
                    line
                    and probe.textbbox((0, 0), line + char, font=font, stroke_width=stroke)[2]
                    > max_width
                ):
                    lines.append(line)
                    line = ""
                line += char
            lines.append(line)
        wrapped = "\n".join(lines)
        spacing = max(1, size // 5)
        box = probe.multiline_textbbox(
            (0, 0), wrapped, font=font, spacing=spacing, stroke_width=stroke
        )
        bw, bh = max(1, box[2] - box[0]), max(1, box[3] - box[1])
        angle = math.radians(operation["rotation"])
        rw = math.ceil(abs(math.cos(angle)) * bw + abs(math.sin(angle)) * bh) + 2
        rh = math.ceil(abs(math.sin(angle)) * bw + abs(math.cos(angle)) * bh) + 2
        if rw > max_width or rh > max_height:
            if size == 10:
                raise ValueError(
                    "Caption cannot fit legibly. Shorten the text or use a larger image."
                )
            size = max(10, int(size * 0.85))
            continue
        layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        ImageDraw.Draw(layer).multiline_text(
            (-box[0], -box[1]),
            wrapped,
            font=font,
            fill=fill,
            spacing=spacing,
            stroke_width=stroke,
            stroke_fill=outline,
            align="center",
        )
        layer = layer.rotate(operation["rotation"], resample=Image.Resampling.BICUBIC, expand=True)
        if layer.width <= max_width and layer.height <= max_height:
            break
        if size == 10:
            raise ValueError("Caption cannot fit legibly. Shorten the text or use a larger image.")
        size = max(10, int(size * 0.85))
    vertical, horizontal = operation["position"].split("_")
    x = {
        "left": margin,
        "center": (width - layer.width) // 2,
        "right": width - margin - layer.width,
    }[horizontal]
    y = {
        "top": margin,
        "center": (height - layer.height) // 2,
        "bottom": height - margin - layer.height,
    }[vertical]
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.alpha_composite(layer, (x, y))
    buffer = io.BytesIO()
    canvas.save(buffer, "PNG")
    return buffer.getvalue()


async def append(graph, source, width, height, operations, upload):
    """Render only font layers locally; crop/resize/compositing execute in ComfyUI."""
    planned = plan(operations, width, height)
    # Render all text layers before uploading anything to catch invalid captions early.
    overlays = {}
    for index, (operation, w, h) in enumerate(planned):
        if operation["operation"] == "text":
            overlays[index] = await asyncio.to_thread(text_overlay, w, h, operation)
    next_id = 1000
    for index, (operation, w, h) in enumerate(planned):
        key = str(next_id)
        if operation["operation"] == "crop":
            graph[key] = {
                "class_type": "ImageCrop",
                "inputs": {
                    "image": source,
                    **{k: operation[k] for k in ("x", "y", "width", "height")},
                },
            }
            source = [key, 0]
            next_id += 1
        elif operation["operation"] == "resize":
            graph[key] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": source,
                    "width": w,
                    "height": h,
                    "upscale_method": "lanczos",
                    "crop": "disabled",
                },
            }
            source = [key, 0]
            next_id += 1
        else:
            filename = await upload(overlays[index], f"caption-{secrets.token_hex(12)}.png")
            invert, composite = str(next_id + 1), str(next_id + 2)
            graph[key] = {"class_type": "LoadImage", "inputs": {"image": filename}}
            # ComfyUI LoadImage returns 1-alpha as its MASK.
            graph[invert] = {"class_type": "InvertMask", "inputs": {"mask": [key, 1]}}
            graph[composite] = {
                "class_type": "ImageCompositeMasked",
                "inputs": {
                    "destination": source,
                    "source": [key, 0],
                    "mask": [invert, 0],
                    "x": 0,
                    "y": 0,
                    "resize_source": False,
                },
            }
            source = [composite, 0]
            next_id += 3
    graph["12"]["inputs"]["images"] = source
    return graph


def existing(filename):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": filename}},
        "12": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0], "filename_prefix": "chatbot-processed"},
        },
    }

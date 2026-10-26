"""Core-node Klein workflows, using explicit API inputs rather than string placeholders."""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(
    prompt: str,
    seed: int,
    width: int,
    height: int,
    images: list[str],
    model: str,
    encoder: str,
    vae: str,
) -> dict:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise ValueError("Provide a prompt of 1–12000 characters.")
    if width % 16 or height % 16 or not 256 <= width <= 1536 or not 256 <= height <= 1536:
        raise ValueError("Dimensions must be multiples of 16 between 256 and 1536.")
    if len(images) > 3:
        raise ValueError("At most three reference images are supported.")
    graph = copy.deepcopy(json.loads((ROOT / "workflows/klein-text-to-image.api.json").read_text()))
    graph["1"]["inputs"]["unet_name"] = model
    graph["2"]["inputs"]["clip_name"] = encoder
    graph["3"]["inputs"]["vae_name"] = vae
    graph["4"]["inputs"]["text"] = prompt
    graph["5"]["inputs"]["noise_seed"] = seed
    for node in ("8", "9"):
        graph[node]["inputs"].update(width=width, height=height)
    # Klein distilled uses CFG=1: BasicGuider needs only positive conditioning.
    # Chain one ReferenceLatent per image, in user-supplied order.
    previous = ["4", 0]
    for index, image in enumerate(images):
        base = 20 + index * 4
        graph[str(base)] = {"class_type": "LoadImage", "inputs": {"image": image}}
        graph[str(base + 1)] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": [str(base), 0],
                "upscale_method": "nearest-exact",
                "megapixels": 1.0,
                "resolution_steps": 1,
            },
        }
        graph[str(base + 2)] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [str(base + 1), 0], "vae": ["3", 0]},
        }
        graph[str(base + 3)] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": previous, "latent": [str(base + 2), 0]},
        }
        previous = [str(base + 3), 0]
    graph["6"]["inputs"]["conditioning"] = previous
    return graph

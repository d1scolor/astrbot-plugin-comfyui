import io

from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image

from engine.client import ComfyClient


async def test_http_contract_upload_submit_history_download():
    seen = {}
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, "PNG")

    async def upload(request):
        parts = await request.multipart()
        image = await parts.next()
        seen["filename"] = image.filename
        seen["image"] = await image.read()
        return web.json_response({"name": "uploaded.png", "subfolder": "refs"})

    async def prompt(request):
        seen["prompt"] = await request.json()
        return web.json_response({"prompt_id": "job-server"})

    async def history(request):
        return web.json_response({"job-server": {"status": {"completed": True}, "outputs": {}}})

    async def view(request):
        seen["query"] = dict(request.query)
        return web.Response(body=buffer.getvalue(), content_type="image/png")

    app = web.Application()
    app.router.add_post("/upload/image", upload)
    app.router.add_post("/prompt", prompt)
    app.router.add_get("/history/{prompt_id}", history)
    app.router.add_get("/view", view)
    async with TestServer(app) as server:
        client = ComfyClient(str(server.make_url("/")).rstrip("/"))
        await client.start()
        try:
            assert await client.upload(buffer.getvalue(), "input.png") == "refs/uploaded.png"
            assert await client.submit({"node": {}}, "plugin-job") == "job-server"
            assert seen["prompt"]["extra_data"] == {"direct_job_id": "plugin-job"}
            assert (await client.history("job-server"))["status"]["completed"]
            result = await client.image({"filename": "a.png", "subfolder": "", "type": "output"})
            assert Image.open(io.BytesIO(result)).size == (16, 16)
            assert seen["query"]["type"] == "output"
        finally:
            await client.close()

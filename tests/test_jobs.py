import asyncio
import json
import time

import pytest

from engine.jobs import Jobs


class Backend:
    def __init__(self):
        self.submissions = 0
        self.ready = asyncio.Event()
        self.ready.set()
        self.fail = False

    async def submit(self, graph, job):
        self.submissions += 1
        return job

    async def history(self, prompt_id):
        await self.ready.wait()
        if self.fail:
            return {"status": {"status_str": "error"}}
        return {
            "status": {"completed": True},
            "outputs": {"12": {"images": [{"filename": "image.png", "type": "output"}]}},
        }

    async def image(self, descriptor):
        return b"mock validated image"


async def settle(jobs):
    await asyncio.gather(*list(jobs.tasks.values()))


async def test_concurrent_groups_followup_and_duplicate(tmp_path):
    backend = Backend()
    jobs = Jobs(tmp_path, backend)
    sent = []

    def sender(origin):
        async def send(paths, text):
            sent.append((origin, paths))

        return send

    first = await jobs.enqueue("group-a", "alice", "message-a", ["fox"], {}, sender("group-a"))
    second = await jobs.enqueue("group-b", "bob", "message-b", ["cat"], {}, sender("group-b"))
    duplicate = await jobs.enqueue("group-a", "alice", "message-a", ["fox"], {}, sender("wrong"))
    assert duplicate["id"] == first["id"]
    await settle(jobs)
    assert backend.submissions == 2
    assert [s[0] for s in sent] == ["group-a", "group-b"]
    assert first["id"] in sent[0][1][0].name
    assert second["id"] in sent[1][1][0].name
    assert jobs.get(first["id"])["state"] == "delivered"
    with pytest.raises(ValueError):
        jobs.owned(first["id"], "group-a", "bob")
    with pytest.raises(ValueError):
        jobs.owned(first["id"], "group-b", "alice")
    await jobs.close()


async def test_failed_delivery_requires_explicit_retry(tmp_path):
    jobs = Jobs(tmp_path, Backend())

    async def broken(*args):
        raise RuntimeError("network")

    row = await jobs.enqueue("private", "a", "m", [], {}, broken)
    await settle(jobs)
    assert jobs.get(row["id"])["state"] == "delivery_unknown"
    sent = []

    async def good(paths, text):
        sent.append(paths)

    await jobs.deliver(row["id"], good)
    assert not sent
    await jobs.deliver(row["id"], good, allow_retry=True)
    await jobs.deliver(row["id"], good, allow_retry=True)
    assert len(sent) == 1
    await jobs.close()


async def test_restart_resume_and_uncertain_states(tmp_path):
    backend = Backend()
    backend.ready.clear()
    jobs = Jobs(tmp_path, backend)
    row = await jobs.enqueue("group", "a", "m", [], {}, None)
    await asyncio.sleep(0)
    assert jobs.get(row["id"])["state"] == "running"
    await jobs.close()
    backend.ready.set()
    jobs = Jobs(tmp_path, backend)
    await jobs.recover()
    await settle(jobs)
    assert backend.submissions == 1
    assert jobs.get(row["id"])["state"] == "generated"
    jobs.update(row["id"], state="sending")
    await jobs.close()
    jobs = Jobs(tmp_path, backend)
    await jobs.recover()
    assert jobs.get(row["id"])["state"] == "delivery_unknown"
    jobs.update(row["id"], state="submitting")
    await jobs.recover()
    assert jobs.get(row["id"])["state"] == "submission_unknown"
    await jobs.close()


async def test_backend_failure_and_timeout(tmp_path):
    backend = Backend()
    backend.fail = True
    jobs = Jobs(tmp_path, backend)
    row = await jobs.enqueue("group", "a", "m", [], {}, None)
    await settle(jobs)
    assert jobs.get(row["id"])["state"] == "failed"
    backend.fail = False
    row = await jobs.enqueue("group", "a", "m2", [], {}, None)
    jobs.update(row["id"], created=time.time() - 1000)
    await settle(jobs)
    assert jobs.get(row["id"])["state"] == "timed_out"
    await jobs.close()


async def test_backpressure_and_delivery_lock(tmp_path):
    backend = Backend()
    backend.ready.clear()
    jobs = Jobs(tmp_path, backend, queue_limit=1)
    row = await jobs.enqueue("g", "a", "m", [], {}, None)
    with pytest.raises(ValueError, match="full"):
        await jobs.enqueue("g", "b", "n", [], {}, None)
    backend.ready.set()
    await settle(jobs)
    sent = []

    async def send(paths, text):
        await asyncio.sleep(0)
        sent.append(paths)

    await asyncio.gather(jobs.deliver(row["id"], send), jobs.deliver(row["id"], send))
    assert len(sent) == 1
    assert len(json.loads(jobs.get(row["id"])["outputs"])) == 1
    await jobs.close()

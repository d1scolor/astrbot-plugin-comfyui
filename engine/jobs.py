"""Durable jobs. Never automatically retry uncertain submission or delivery."""

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

from .images import Images

ACTIVE = {"queued", "submitting", "running", "generated", "sending"}


class Jobs:
    def __init__(
        self, directory: Path, client, timeout=600, poll=1.0, queue_limit=8, retention_days=30
    ):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.client = directory, client
        self.timeout, self.poll, self.queue_limit = timeout, poll, queue_limit
        self.db = sqlite3.connect(directory / "jobs.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, dedup TEXT UNIQUE, origin TEXT, owner TEXT,
          state TEXT, graph TEXT, prompt_id TEXT, created REAL, outputs TEXT,
          error TEXT DEFAULT '')""")
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(jobs)")}
        if "description" not in columns:
            self.db.execute("ALTER TABLE jobs ADD COLUMN description TEXT DEFAULT ''")
        if "inputs" not in columns:
            self.db.execute("ALTER TABLE jobs ADD COLUMN inputs TEXT DEFAULT '[]'")
        self.db.commit()
        self.images = Images(directory, self.db, retention_days)
        self.images.prune()
        self.tasks = {}
        self.lock = asyncio.Lock()
        self.worker = asyncio.Semaphore(1)
        self.closed = False

    def update(self, job, **fields):
        self.db.execute(
            "UPDATE jobs SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
            (*fields.values(), job),
        )
        self.db.commit()

    def get(self, job):
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
        return dict(row) if row else None

    def owned(self, job, origin, owner):
        row = self.get(job)
        if not row or row["origin"] != origin or row["owner"] != owner:
            raise ValueError("Job not found in your conversation.")
        return row

    def spawn(self, job, sender=None):
        if job not in self.tasks:
            task = asyncio.create_task(self.run(job, sender))
            self.tasks[job] = task
            task.add_done_callback(lambda _: self.tasks.pop(job, None))

    def output_images(self, row):
        result = []
        for index, name in enumerate(json.loads(row["outputs"])):
            path = self.directory / name
            if not path.is_file():
                continue
            source_key = f"{row['id']}:{index}"
            # Do not renew expired handles when status is checked.
            existing = self.db.execute(
                "SELECT * FROM images WHERE origin=? AND owner=? AND source='generated' AND source_key=?",
                (row["origin"], row["owner"], source_key),
            ).fetchone()
            if existing:
                try:
                    image = self.images.get(existing["id"], row["origin"], row["owner"])
                except ValueError:
                    continue
            elif time.time() - row["created"] < self.images.ttl:
                image = self.images.add(
                    row["origin"],
                    row["owner"],
                    "generated",
                    source_key,
                    path.read_bytes(),
                    f"Generated: {row.get('description', '')[:220]} (output {index + 1})",
                )
            else:
                continue
            result.append(self.images.public(image))
        return result

    async def recover(self):
        # A crash between request and acknowledgement cannot be resolved safely by resubmission.
        self.db.execute("UPDATE jobs SET state='submission_unknown' WHERE state='submitting'")
        self.db.execute("UPDATE jobs SET state='delivery_unknown' WHERE state='sending'")
        self.db.commit()
        for row in self.db.execute(
            "SELECT * FROM jobs WHERE outputs!='[]' ORDER BY created ASC"
        ).fetchall():
            self.output_images(dict(row))
        for row in self.db.execute(
            "SELECT id FROM jobs WHERE state IN ('queued','running')"
        ).fetchall():
            self.spawn(row["id"])

    async def enqueue(self, origin, owner, message_id, signature, graph, sender, input_ids=()):
        if not origin or not owner or not message_id:
            raise ValueError("This adapter did not supply a stable request identity.")
        dedup = hashlib.sha256(
            json.dumps([origin, owner, message_id, signature]).encode()
        ).hexdigest()
        async with self.lock:
            previous = self.db.execute("SELECT id FROM jobs WHERE dedup=?", (dedup,)).fetchone()
            if previous:
                return self.get(previous["id"])
            count = self.db.execute(
                "SELECT count(*) FROM jobs WHERE state IN "
                "('queued','submitting','running','generated','sending')"
            ).fetchone()[0]
            if count >= self.queue_limit:
                raise ValueError("Image queue is full; try again after existing jobs finish.")
            own = self.db.execute(
                "SELECT count(*) FROM jobs WHERE origin=? AND owner=? AND state IN "
                "('queued','submitting','running')",
                (origin, owner),
            ).fetchone()[0]
            if own >= 2:
                raise ValueError("You already have two active image jobs in this conversation.")
            job = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO jobs(id,dedup,origin,owner,state,graph,created,outputs) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (job, dedup, origin, owner, "queued", json.dumps(graph), time.time(), "[]"),
            )
            description = signature[0] if signature and isinstance(signature[0], str) else ""
            self.db.execute(
                "UPDATE jobs SET description=?, inputs=? WHERE id=?",
                (description[:300], json.dumps(list(input_ids)), job),
            )
            self.db.commit()
            self.spawn(job, sender)
            return self.get(job)

    async def run(self, job, sender=None):
        try:
            async with self.worker:
                row = self.get(job)
                if row["state"] not in ("queued", "running"):
                    return
                if time.time() - row["created"] >= self.timeout:
                    self.update(job, state="timed_out", error="Job deadline exceeded.")
                    return
                if row["state"] == "queued":
                    self.update(job, state="submitting")
                    try:
                        prompt_id = await self.client.submit(json.loads(row["graph"]), job)
                    except Exception:
                        self.update(
                            job,
                            state="submission_unknown",
                            error="Submission not confirmed; not retried.",
                        )
                        return
                    self.update(job, state="running", prompt_id=prompt_id)
                row = self.get(job)
                while time.time() - row["created"] < self.timeout:
                    history = await self.client.history(row["prompt_id"])
                    if history:
                        status = history.get("status", {})
                        if status.get("status_str") == "error":
                            raise RuntimeError("execution_failed")
                        if status.get("completed"):
                            descriptors = [
                                image
                                for output in history.get("outputs", {}).values()
                                for image in output.get("images", [])
                                if image.get("type") == "output"
                            ]
                            if not descriptors:
                                raise RuntimeError("no_image_output")
                            if len(descriptors) > 4:
                                raise RuntimeError("too_many_outputs")
                            paths = []
                            for index, descriptor in enumerate(descriptors):
                                data = await self.client.image(descriptor)
                                path = self.directory / f"{job}-{index}.png"
                                path.write_bytes(data)
                                paths.append(path.name)
                            self.update(
                                job, state="generated", outputs=json.dumps(paths), graph="{}"
                            )
                            self.output_images(self.get(job))
                            break
                    await asyncio.sleep(self.poll)
                else:
                    self.update(
                        job,
                        state="timed_out",
                        error="Monitoring timed out; server job may still run.",
                    )
                    return
            if sender:
                await self.deliver(job, sender)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.update(job, state="failed", error="ComfyUI execution or output retrieval failed.")
        finally:
            row = self.get(job)
            if sender and row["state"] in ("failed", "timed_out", "submission_unknown"):
                try:
                    await sender(
                        [], f"Image job {job}: {row['state']}. Use comfy_image_status for details."
                    )
                except Exception:
                    pass

    async def deliver(self, job, sender, allow_retry=False):
        async with self.lock:
            row = self.get(job)
            eligible = {"generated", "delivery_failed"}
            if allow_retry:
                eligible.add("delivery_unknown")
            if row["state"] not in eligible:
                return
            paths = [self.directory / name for name in json.loads(row["outputs"])]
            self.update(job, state="sending")
        try:
            await sender(paths, "")
        except Exception:
            # Transport could have accepted a message before the exception.
            self.update(
                job,
                state="delivery_unknown",
                error="Delivery not confirmed; explicit retry may duplicate.",
            )
        else:
            self.update(job, state="delivered", error="")

    async def close(self):
        self.closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.db.close()

import time

import pytest
from test_jobs import Backend

from engine.jobs import Jobs


def test_handle_ownership_persistence_and_expiry(tmp_path):
    jobs = Jobs(tmp_path, Backend(), retention_days=1)
    images = jobs.images
    row = images.add("group-a", "alice", "attachment", "message:0", b"validated", "User image")
    assert row["id"].startswith("img_")
    assert (
        images.add("group-a", "alice", "attachment", "message:0", b"validated", "User image")["id"]
        == row["id"]
    )
    assert images.data(row["id"], "group-a", "alice") == b"validated"
    assert "filename" not in images.public(row)
    for origin, owner in [("group-b", "alice"), ("group-a", "bob")]:
        with pytest.raises(ValueError):
            images.get(row["id"], origin, owner)
        assert images.recent(origin, owner) == []
    jobs.db.close()
    jobs = Jobs(tmp_path, Backend())
    assert jobs.images.get(row["id"], "group-a", "alice")["id"] == row["id"]
    jobs.db.execute("UPDATE images SET expires=?", (time.time() - 1,))
    jobs.db.commit()
    with pytest.raises(ValueError):
        jobs.images.get(row["id"], "group-a", "alice")
    jobs.images.prune()
    assert not (jobs.images.directory / row["filename"]).exists()
    assert jobs.images.recent("group-a", "alice") == []
    jobs.db.close()


def test_migrate_existing_jobs_and_paginate_images(tmp_path):
    import json
    import sqlite3

    db = sqlite3.connect(tmp_path / "jobs.sqlite3")
    db.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, dedup TEXT UNIQUE, origin TEXT, owner TEXT, "
        "state TEXT, graph TEXT, prompt_id TEXT, created REAL, outputs TEXT, error TEXT DEFAULT '')"
    )
    db.execute(
        "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            "legacy",
            "dedup",
            "group",
            "alice",
            "delivered",
            "{}",
            "server",
            time.time(),
            json.dumps(["legacy-0.png"]),
            "",
        ),
    )
    db.commit()
    db.close()
    (tmp_path / "legacy-0.png").write_bytes(b"validated")
    jobs = Jobs(tmp_path, Backend())
    outputs = jobs.output_images(jobs.get("legacy"))
    assert len(outputs) == 1
    assert jobs.get("legacy")["inputs"] == "[]"
    jobs.images.add("group", "alice", "attachment", "message:0", b"validated", "Input")
    first = jobs.images.recent("group", "alice", limit=1)
    second = jobs.images.recent("group", "alice", limit=1, offset=1)
    assert first[0]["id"] != second[0]["id"]
    jobs.db.close()

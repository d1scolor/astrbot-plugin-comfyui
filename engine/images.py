"""Persistent, owner-scoped media handles. Paths and source URLs never reach the model."""

import hashlib
import time
import uuid


class Images:
    def __init__(self, directory, db, retention_days=30):
        self.directory = directory / "images"
        self.directory.mkdir(exist_ok=True)
        self.db = db
        self.ttl = retention_days * 86400
        db.execute("""CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY, origin TEXT, owner TEXT, source TEXT, source_key TEXT,
            label TEXT, filename TEXT, created REAL, expires REAL,
            UNIQUE(origin, owner, source, source_key))""")
        db.execute(
            "CREATE TABLE IF NOT EXISTS image_inspections (image_id TEXT PRIMARY KEY, checked REAL)"
        )
        db.commit()

    def prune(self):
        rows = self.db.execute(
            "SELECT id, filename FROM images WHERE expires<=?", (time.time(),)
        ).fetchall()
        for row in rows:
            (self.directory / row["filename"]).unlink(missing_ok=True)
            self.db.execute("DELETE FROM images WHERE id=?", (row["id"],))
        self.db.execute(
            "DELETE FROM image_inspections WHERE image_id NOT IN (SELECT id FROM images)"
        )
        self.db.commit()

    def mark_inspected(self, image_id, origin, owner):
        self.get(image_id, origin, owner)
        self.db.execute(
            "INSERT OR REPLACE INTO image_inspections VALUES(?,?)", (image_id, time.time())
        )
        self.db.commit()

    def is_inspected(self, image_id, origin, owner):
        self.get(image_id, origin, owner)
        return (
            self.db.execute(
                "SELECT 1 FROM image_inspections WHERE image_id=?", (image_id,)
            ).fetchone()
            is not None
        )

    def add(self, origin, owner, source, source_key, data, label):
        if not origin or not owner:
            raise ValueError("Image ownership is unavailable.")
        row = self.db.execute(
            "SELECT * FROM images WHERE origin=? AND owner=? AND source=? AND source_key=?",
            (origin, owner, source, source_key),
        ).fetchone()
        if row and row["expires"] > time.time() and (self.directory / row["filename"]).is_file():
            return dict(row)
        if row:
            (self.directory / row["filename"]).unlink(missing_ok=True)
            self.db.execute("DELETE FROM images WHERE id=?", (row["id"],))
        image_id = "img_" + uuid.uuid4().hex
        filename = image_id + ".png"
        (self.directory / filename).write_bytes(data)
        now = time.time()
        self.db.execute(
            "INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?)",
            (
                image_id,
                origin,
                owner,
                source,
                source_key,
                label[:300],
                filename,
                now,
                now + self.ttl,
            ),
        )
        self.db.commit()
        return dict(self.db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone())

    def get(self, image_id, origin, owner):
        row = self.db.execute(
            "SELECT * FROM images WHERE id=? AND origin=? AND owner=? AND expires>?",
            (image_id, origin, owner, time.time()),
        ).fetchone()
        if not row or not (self.directory / row["filename"]).is_file():
            raise ValueError(
                "Image not found, expired or unavailable in your conversation. List images or reattach it."
            )
        return dict(row)

    def data(self, image_id, origin, owner):
        row = self.get(image_id, origin, owner)
        return (self.directory / row["filename"]).read_bytes()

    def recent(self, origin, owner, limit=8, offset=0):
        rows = self.db.execute(
            "SELECT * FROM images WHERE origin=? AND owner=? AND expires>? "
            "ORDER BY created DESC, id DESC LIMIT ? OFFSET ?",
            (origin, owner, time.time(), limit, offset),
        ).fetchall()
        return [dict(row) for row in rows if (self.directory / row["filename"]).is_file()]

    @staticmethod
    def public(row):
        return {key: row[key] for key in ("id", "source", "label", "created", "expires")}

    @staticmethod
    def content_key(data):
        return hashlib.sha256(data).hexdigest()

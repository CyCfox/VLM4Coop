"""Persistent teacher labels; SQLite access stays on the calling thread."""
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
import numpy as np
from vlm4coop.teachers.interfaces import TeacherResult

class ProbabilityCache:
    def __init__(self, path, identity, max_bytes=2_000_000_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.max_bytes = max_bytes
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("CREATE TABLE IF NOT EXISTS predictions (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()
        self.hits = self.misses = self.writes = self.capped = 0

    def key(self, image, n_agents):
        rgb = np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
        header = json.dumps({"identity": self.identity, "n_agents": n_agents, "shape": rgb.shape}, sort_keys=True)
        return sha256(header.encode() + rgb.tobytes()).hexdigest()

    def get(self, key):
        row = self.db.execute("SELECT value FROM predictions WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        value = json.loads(row[0])
        return TeacherResult(np.asarray(value["probs"], dtype=np.float32), value["raw"], value["repaired"], cache_hit=True)

    def put(self, key, result, replace=False):
        if result.source != 'teacher':
            return False  # masked/controller defaults are not reusable teacher labels
        encoded = json.dumps({"raw": result.raw, "probs": result.probs.tolist(), "repaired": result.repaired}, allow_nan=False)
        # The page count includes free pages; budget conservatively and disable new entries.
        allocated = self.db.execute("PRAGMA page_count").fetchone()[0] * self.db.execute("PRAGMA page_size").fetchone()[0]
        if allocated + len(encoded.encode()) + 8192 > self.max_bytes:
            self.capped += 1
            return False
        statement = "INSERT OR REPLACE INTO predictions VALUES (?,?)" if replace else "INSERT OR IGNORE INTO predictions VALUES (?,?)"
        self.db.execute(statement, (key, encoded))
        self.db.commit()
        self.writes += 1
        return True

    def close(self):
        self.db.close()


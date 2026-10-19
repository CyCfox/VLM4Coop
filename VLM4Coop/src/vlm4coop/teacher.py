"""Single-image, all-agent local teacher. Text-declared probabilities, not logits."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re
import sqlite3
import time
import numpy as np
from PIL import Image

ACTIONS = ("NONE", "NORTH", "SOUTH", "WEST", "EAST", "LOAD")


SYSTEM_PROMPT = "You are a concise cooperative game controller. Spend reasoning on joint action selection. Do not narrate the image or repeat rules. Finish with the requested probability JSON."


def joint_prompt(n_agents):
    return f"""Choose one simultaneous joint move for all {n_agents} agents in the attached LBF grid.
Blue circles A0..A{n_agents-1} are agents; green squares F are food; L is level. Rows increase down, columns right.
Action indices: 0=NONE(wait), 1=NORTH(up), 2=SOUTH(down), 3=WEST(left), 4=EAST(right), 5=LOAD.
LOAD works only for agents immediately adjacent to the SAME food, and ONLY if the sum of levels of those adjacent agents choosing LOAD simultaneously reaches its level.
A distant agent cannot LOAD food. An agent alone cannot load food above its level. Move partners into adjacent cells before loading together. Avoid colliding moves.
The goal is to collect ALL food efficiently over future steps, not to collect it immediately this step. Choose useful approaches if no food can be collected now.
Use your brief reasoning to decide actions for the whole team, WITHOUT transcribing the grid or re-explaining these rules.
Then return ONLY one compact JSON object: {{"agents":[{{"id":0,"probs":[six decimal numbers]}},...]}}.
Include exactly IDs 0..{n_agents-1}, exactly six probabilities each in the action order above. Each value must be in [0,1], with at most three decimal places; each row sum exactly 1, never all zero.
Example of syntax ONLY (choose your own probabilities): {{"id":0,"probs":[0.05,0.15,0.1,0.1,0.5,0.1]}}.
Express confidence over the available moves; values are not action numbers."""


@dataclass
class TeacherResult:
    probs: np.ndarray
    raw: str
    repaired: bool
    latency_seconds: float = 0.0
    generated_tokens: int = 0
    cache_hit: bool = False
    source: str = 'teacher'
    format_attempts: int = 1
    decision_id: str = ''


def parse_probabilities(raw, n_agents):
    # Legacy local-Qwen/archive parser. Active API experiments use the strict,
    # role-specific parser in response_protocol.py.
    # Thinking can contain drafts; parse only the final answer after </think>.
    answer = raw.rsplit("</think>", 1)[-1].strip()
    decoder = json.JSONDecoder()
    obj = None
    obj_span = (-1, -1)
    offset = 0
    # Content may contain prose, fenced JSON and draft objects. Decode whole
    # objects and select the final all-agent object rather than a reasoning draft.
    while offset < len(answer):
        start = answer.find("{", offset)
        if start < 0:
            break
        try:
            candidate, end = decoder.raw_decode(answer, start)
        except ValueError:
            offset = start + 1
            continue
        if isinstance(candidate, dict) and "agents" in candidate:
            obj = candidate
            obj_span = (start, end)
        offset = end
    syntax_repaired = False
    markers = list(re.finditer(r'"agents"\s*:\s*\[', answer))
    if markers and not (obj_span[0] <= markers[-1].start() < obj_span[1]):
        # Observed API response: every agent object and six-value row was
        # complete, but the enclosing agents list lacked its closing bracket.
        # Recover only complete JSON row objects from the final agents list;
        # never infer an absent ID, action entry or probability value.
        tail = answer[markers[-1].end():]
        recovered, cursor = [], 0
        while cursor < len(tail):
            start = tail.find('{', cursor)
            if start < 0:
                break
            try:
                row, end = decoder.raw_decode(tail, start)
            except ValueError:
                raise ValueError('Incomplete agent probability object') from None
            if not isinstance(row, dict) or 'id' not in row or 'probs' not in row:
                raise ValueError('Cannot identify complete agent probability rows')
            recovered.append(row)
            cursor = end
        obj = {'agents': recovered}
        syntax_repaired = True
    if obj is None or not isinstance(obj["agents"], list) or len(obj["agents"]) != n_agents:
        raise ValueError("Missing all-agent JSON")
    rows = obj["agents"]
    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(ids) != n_agents or any(type(i) is not int for i in ids) or sorted(ids) != list(range(n_agents)):
        raise ValueError("Missing, duplicate or invalid agent IDs")
    probs, repaired = [], syntax_repaired
    for row in sorted(rows, key=lambda item: item["id"]):
        values = row.get("probs")
        if not isinstance(values, list) or not values:
            raise ValueError("Cannot identify action entries")
        numeric = np.asarray([float(v) if type(v) in (int, float) else np.nan for v in values])
        if not np.isfinite(numeric).any():
            raise ValueError("No identifiable finite maximum")
        maximum = int(np.nanargmax(numeric))
        confidence = numeric[maximum]
        if maximum >= 6:
            raise ValueError("Maximum has no legal action index; cannot preserve it")
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Maximum is not a legal probability; cannot preserve it")
        p = numeric.copy() if len(numeric) == 6 else np.empty(6, dtype=float)
        valid = len(numeric) == 6 and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        if not valid or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
            p[:] = (1 - confidence) / 5
            p[maximum] = confidence
            repaired = True
        p = p / p.sum()  # floating-point roundoff only
        probs.append(p)
    return np.asarray(probs, dtype=np.float32), repaired


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


class LocalQwenTeacher:
    def __init__(self, config, cache, audit_path):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
        self.config, self.cache = config, cache
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config["model_path"], local_files_only=True, dtype=getattr(torch, config["dtype"]),
            attn_implementation="eager").to(config["device"]).eval()
        self.calls = 0
        self.generate_calls = 0
        self.processor.tokenizer.padding_side = "left"
        from vlm4coop.structured_output import build_tokenizer_data
        self.format_tokenizer_data = build_tokenizer_data(self.processor.tokenizer)

    def predict(self, image, n_agents, use_cache=True):
        import torch
        image = image.convert("RGB")
        key = self.cache.key(image, n_agents)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "image", "image": image},
                     {"type": "text", "text": joint_prompt(n_agents)}]}]
        for attempt in range(1):
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.config["enable_thinking"])
            if self.config["enable_thinking"] and not prompt.endswith("<think>\n"):
                raise ValueError("Thinking-mode template did not open a reasoning block")
            # Render custom Jinja flags before processor tokenization, avoiding
            # the processor's unrelated unknown-keyword warning.
            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
            torch.cuda.synchronize()
            start = time.monotonic()
            from vlm4coop.thinking_budget import ThinkingBudget
            closing_ids = self.processor.tokenizer.encode("</think>", add_special_tokens=False)
            if len(closing_ids) != 1:
                raise ValueError("Thinking budget requires a single closing-think token")
            processor = ThinkingBudget(inputs["input_ids"].shape[-1], closing_ids[0],
                                       self.config["thinking_budget_tokens"])
            from vlm4coop.structured_output import MatrixJsonOutput
            json_output = MatrixJsonOutput(self.format_tokenizer_data, inputs["input_ids"].shape[-1],
                closing_ids[0], n_agents, self.config["probability_decimals"])
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, do_sample=False,
                                              max_new_tokens=self.config["max_new_tokens"],
                                              logits_processor=[processor, json_output])
            self.generate_calls += 1
            torch.cuda.synchronize()
            latency = time.monotonic() - start
            token_ids = outputs[0, inputs["input_ids"].shape[-1]:]
            raw = self.processor.decode(token_ids, skip_special_tokens=False)
            self.calls += 1
            event = {"key": key, "attempt": attempt, "raw": raw, "latency_seconds": latency,
                     "generated_tokens": len(token_ids), "n_agents": n_agents,
                     "image_count": 1, "thinking_enabled": self.config["enable_thinking"],
                     "thinking_completed": "</think>" in raw,
                     "thinking_budget_tokens": self.config["thinking_budget_tokens"]}
            try:
                if self.config["enable_thinking"] and "</think>" not in raw:
                    raise ValueError("Thinking truncated before final probability output")
                probs, repaired = parse_probabilities(raw, n_agents)
                event.update(valid=True, repaired=repaired)
            except ValueError as exc:
                event.update(valid=False, error=str(exc))
            with self.audit_path.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if event["valid"]:
                result = TeacherResult(probs, raw, repaired, latency, len(token_ids))
                if use_cache:
                    self.cache.put(key, result)
                return result
        raise ValueError(f"Teacher output unrecoverable; raw output saved to {self.audit_path}")

"""Unchanged single-image local Qwen backend (historical LBF protocol)."""
from pathlib import Path
import json
import time
from vlm4coop.teachers.interfaces import TeacherResult
from vlm4coop.teachers.legacy_parser import parse_probabilities
from vlm4coop.envs.lbf.prompts import SYSTEM_PROMPT, joint_prompt

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
        from vlm4coop.teachers.structured_output import build_tokenizer_data
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
            from vlm4coop.teachers.thinking_budget import ThinkingBudget
            closing_ids = self.processor.tokenizer.encode("</think>", add_special_tokens=False)
            if len(closing_ids) != 1:
                raise ValueError("Thinking budget requires a single closing-think token")
            processor = ThinkingBudget(inputs["input_ids"].shape[-1], closing_ids[0],
                                       self.config["thinking_budget_tokens"])
            from vlm4coop.teachers.structured_output import MatrixJsonOutput
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


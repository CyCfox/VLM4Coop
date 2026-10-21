"""Construct teachers without depending on the training loop."""
from pathlib import Path
import json
from vlm4coop.utils.paths import PROJECT
from vlm4coop.envs.registry import get_adapter
from vlm4coop.envs.lbf.rendering import RENDERER_VERSION
from vlm4coop.envs.lbf.prompts import joint_prompt, SYSTEM_PROMPT
from vlm4coop.teachers.cache import ProbabilityCache
from vlm4coop.teachers.local import LocalQwenTeacher

def make_teacher(config, output, teacher_type=LocalQwenTeacher, use_cache=True, purpose='distillation'):
    adapter = get_adapter(config)
    protocol = adapter.teacher_spec()
    if config["teacher"].get("backend") == "openai_api":
        from vlm4coop.teachers.api import OpenCodeTeacher, build_api_cache_identity
        identity = build_api_cache_identity(config['teacher'],protocol)
        cache = (ProbabilityCache(PROJECT / "cache/teacher.sqlite", identity,
                                  max_bytes=config["cache_max_bytes"]) if use_cache else None)
        return OpenCodeTeacher(config["teacher"], output / "teacher_calls.jsonl", cache,purpose=purpose,protocol=protocol)
    if adapter.name != "lbf":
        raise NotImplementedError("Local Qwen currently implements the LBF protocol only")
    model_path = Path(config["teacher"]["model_path"])
    identity = dict(config["teacher"], renderer=RENDERER_VERSION, system_prompt=SYSTEM_PROMPT,
                    model_config=json.loads((model_path / "config.json").read_text()),
                    weights=[(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                             for p in sorted(model_path.glob("*.safetensors"))],
                    prompt_templates={n: joint_prompt(n) for n in (3, 4, 5)})
    cache = (ProbabilityCache(PROJECT / "cache/teacher.sqlite", identity,
                              max_bytes=config["cache_max_bytes"]) if use_cache else None)
    return teacher_type(config["teacher"], cache, output / "teacher_calls.jsonl")


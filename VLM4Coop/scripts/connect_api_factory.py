from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/training.py')
s = p.read_text()
old = 'def make_teacher(config, output, teacher_type=LocalQwenTeacher):\n'
new = '''def make_teacher(config, output, teacher_type=LocalQwenTeacher):
    if config["teacher"].get("backend") == "openai_api":
        from vlm4coop.api_teacher import OpenCodeTeacher
        identity = dict(config["teacher"], renderer=RENDERER_VERSION,
                        system_prompt=SYSTEM_PROMPT,
                        prompt_templates={n: joint_prompt(n) for n in (3, 4, 5)})
        cache = ProbabilityCache(PROJECT / "cache/teacher.sqlite", identity,
                                 max_bytes=config["cache_max_bytes"])
        return OpenCodeTeacher(config["teacher"], output / "teacher_calls.jsonl", cache)
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))

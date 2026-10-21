from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/teacher.py')
s = p.read_text()
s = s.replace('''LOAD collects adjacent food (Manhattan distance 1) only when the summed levels of agents simultaneously choosing LOAD suffice.
Agents cannot occupy the same cell. Coordinate partners, avoid conflicting moves, collect ALL food quickly.''', '''A food is collected only by agents in the four cells immediately adjacent to THAT SAME food (Manhattan distance 1).
ONLY those adjacent agents simultaneously choosing LOAD contribute their levels; their sum must be at least that food's level.
Nonadjacent agents contribute nothing. If levels are insufficient, move partners toward adjacent cells before loading together.
These rules are definitive. Agents cannot occupy the same cell. Coordinate partners, avoid conflicting moves, collect ALL food quickly.''')
s = s.replace('''Return a JSON object with the sole key agents,''', '''Keep your thinking brief (at most 150 tokens): read positions once, choose immediate moves, then finish with JSON. Do not repeat the grid description or debate the rules.
Return a JSON object with the sole key agents,''')
s = s.replace('''torch_dtype=getattr(torch, config["dtype"]),''', '''dtype=getattr(torch, config["dtype"]),''')
s = s.replace('''            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=self.config["enable_thinking"], return_dict=True, return_tensors="pt").to(self.model.device)''', '''            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.config["enable_thinking"])
            if self.config["enable_thinking"] and not prompt.endswith("<think>\\n"):
                raise ValueError("Thinking-mode template did not open a reasoning block")
            # Render custom Jinja flags before processor tokenization, avoiding
            # the processor's unrelated unknown-keyword warning.
            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)''')
s = s.replace('''"generated_tokens": len(token_ids), "n_agents": n_agents}''', '''"generated_tokens": len(token_ids), "n_agents": n_agents,
                     "image_count": 1, "thinking_enabled": self.config["enable_thinking"],
                     "thinking_completed": "</think>" in raw}''')
s = s.replace('''                probs, repaired = parse_probabilities(raw, n_agents)''', '''                if self.config["enable_thinking"] and "</think>" not in raw:
                    raise ValueError("Thinking truncated before final probability output")
                probs, repaired = parse_probabilities(raw, n_agents)''')
p.write_text(s)
p = Path('VLM4Coop/configs/experiments.yaml')
s = p.read_text().replace('max_new_tokens: 2048', 'max_new_tokens: 4096').replace('prompt_version: joint-lbf-v1', 'prompt_version: joint-lbf-v2')
p.write_text(s)
p = Path('VLM4Coop/scripts/teacher_smoke.py')
s = p.read_text().replace('''import time
''', '''import time
from datetime import datetime, timezone, timedelta
''').replace("time.strftime('%Y%m%d_%H%M%S')", "datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')")
s = s.replace('''finally:
    teacher.cache.close()''', '''except BaseException as exc:
    (output / 'status.json').write_text(json.dumps({'status': 'failed', 'error': repr(exc)}))
    raise
finally:
    teacher.cache.close()''')
s = s.replace('''print(json.dumps({'status': 'passed', 'real_teacher_calls': teacher.calls}), flush=True)''', '''(output / 'status.json').write_text(json.dumps({'status': 'passed', 'real_teacher_calls': teacher.calls}))
print(json.dumps({'status': 'passed', 'real_teacher_calls': teacher.calls}), flush=True)''')
p.write_text(s)

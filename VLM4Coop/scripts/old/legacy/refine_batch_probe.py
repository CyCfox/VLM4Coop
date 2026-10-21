from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/training.py')
s = p.read_text().replace('def make_teacher(config, output):', 'def make_teacher(config, output, teacher_type=LocalQwenTeacher):')
s = s.replace('return LocalQwenTeacher(config["teacher"], cache, output / "teacher_calls.jsonl")', 'return teacher_type(config["teacher"], cache, output / "teacher_calls.jsonl")')
p.write_text(s)
p = Path('VLM4Coop/scripts/batch_teacher_smoke.py')
s = p.read_text().replace('teacher = make_teacher(config, output)\nteacher.__class__ = BatchedQwenTeacher', 'teacher = make_teacher(config, output, teacher_type=BatchedQwenTeacher)')
p.write_text(s)
p = Path('VLM4Coop/src/vlm4coop/batched_teacher.py')
s = p.read_text().replace('''                token_ids = outputs[row, inputs['input_ids'].shape[-1]:]
                raw''', '''                token_ids = outputs[row, inputs['input_ids'].shape[-1]:]
                eos = self.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                for end, value in enumerate(token_ids.tolist()):
                    if value in eos:
                        token_ids = token_ids[:end+1]
                        break
                raw''')
s = s.replace('''                    thinking_enabled=True, thinking_budget_tokens=self.config['thinking_budget_tokens'])''', '''                    thinking_enabled=True, thinking_budget_tokens=self.config['thinking_budget_tokens'],
                    actual_thinking_tokens=(token_ids == closing_ids[0]).nonzero()[0].item()
                        if (token_ids == closing_ids[0]).any() else None)''')
p.write_text(s)

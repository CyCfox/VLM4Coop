from pathlib import Path
p = Path('VLM4Coop/configs/experiments.yaml')
s = p.read_text().replace('prompt_version: joint-lbf-v2', 'prompt_version: joint-lbf-v3-budget256')
s = s.replace('  max_new_tokens: 4096', '  max_new_tokens: 2048\n  thinking_budget_tokens: 256')
p.write_text(s)
p = Path('VLM4Coop/src/vlm4coop/teacher.py')
s = p.read_text().replace('''        self.calls = 0''', '''        self.calls = 0
        self.generate_calls = 0
        self.processor.tokenizer.padding_side = "left"''')
s = s.replace('''            with torch.inference_mode():
                outputs = self.model.generate(**inputs, do_sample=False,
                                              max_new_tokens=self.config["max_new_tokens"])''', '''            from vlm4coop.thinking_budget import ThinkingBudget
            closing_ids = self.processor.tokenizer.encode("</think>", add_special_tokens=False)
            if len(closing_ids) != 1:
                raise ValueError("Thinking budget requires a single closing-think token")
            processor = ThinkingBudget(inputs["input_ids"].shape[-1], closing_ids[0],
                                       self.config["thinking_budget_tokens"])
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, do_sample=False,
                                              max_new_tokens=self.config["max_new_tokens"],
                                              logits_processor=[processor])
            self.generate_calls += 1''')
s = s.replace('''"thinking_completed": "</think>" in raw}''', '''"thinking_completed": "</think>" in raw,
                     "thinking_budget_tokens": self.config["thinking_budget_tokens"]}''')
# An identical deterministic retry cannot fix a truncated or malformed answer.
# Keep original evidence and fail immediately after parser's authorized repair.
s = s.replace('for attempt in range(2):', 'for attempt in range(1):')
s = s.replace('Teacher failed twice; raw output saved to', 'Teacher output unrecoverable; raw output saved to')
p.write_text(s)
p = Path('VLM4Coop/configs/experiments.yaml')
s = p.read_text().replace('failure_policy: retry_once_then_fail', 'failure_policy: repair_identifiable_maximum_else_fail')
p.write_text(s)

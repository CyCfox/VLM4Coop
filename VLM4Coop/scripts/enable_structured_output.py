from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/teacher.py')
s = p.read_text().replace('''        self.processor.tokenizer.padding_side = "left"''', '''        self.processor.tokenizer.padding_side = "left"
        from vlm4coop.structured_output import build_tokenizer_data
        self.format_tokenizer_data = build_tokenizer_data(self.processor.tokenizer)''')
s = s.replace('''            with torch.inference_mode():
                outputs''', '''            from vlm4coop.structured_output import MatrixJsonOutput
            json_output = MatrixJsonOutput(self.format_tokenizer_data, inputs["input_ids"].shape[-1],
                closing_ids[0], n_agents, self.config["probability_decimals"])
            with torch.inference_mode():
                outputs''')
s = s.replace('logits_processor=[processor])', 'logits_processor=[processor, json_output])')
s = s.replace('''Each value must be in [0,1], each row sum exactly 1; never output an all-zero row. Express your confidence over the available moves.''', '''Each value must be in [0,1], with at most three decimal places; each row sum exactly 1, never all zero.
Example of syntax ONLY (choose your own probabilities): {{"id":0,"probs":[0.05,0.15,0.1,0.1,0.5,0.1]}}.
Express confidence over the available moves; values are not action numbers.''')
p.write_text(s)
p = Path('VLM4Coop/src/vlm4coop/batched_teacher.py')
s = p.read_text().replace('''            with torch.inference_mode():
                outputs''', '''            from vlm4coop.structured_output import MatrixJsonOutput
            json_output = MatrixJsonOutput(self.format_tokenizer_data, inputs['input_ids'].shape[-1],
                closing_ids[0], n_agents, self.config['probability_decimals'])
            with torch.inference_mode():
                outputs''')
s = s.replace('logits_processor=[budget])', 'logits_processor=[budget, json_output])')
p.write_text(s)
p = Path('VLM4Coop/configs/experiments.yaml')
s = p.read_text().replace('joint-lbf-v4-budget256', 'joint-lbf-v5-budget256-json')
s = s.replace('  parser_version:', '  output_format: matrix-json-regex-v1\n  probability_decimals: 3\n  parser_version:')
p.write_text(s)
p = Path('VLM4Coop/src/vlm4coop/thinking_budget.py')
s = p.read_text().replace('''Optional bounded-thinking generation primitive, NOT enabled by default.

Enabling a limit is a protocol decision pending the user's explicit response.''', '''Bounded-thinking generation primitive, enabled by user authorization.

The user approved a uniform 256-token thinking budget on 2026-10-17.''')
p.write_text(s)
p = Path('VLM4Coop/tests/test_thinking_budget.py')
s = p.read_text().replace('this optional primitive has not been enabled for VLM', 'the user-authorized primitive never supplies actions/probabilities')
p.write_text(s)

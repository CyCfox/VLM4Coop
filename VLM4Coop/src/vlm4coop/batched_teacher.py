"""Legal batch processing: each sample still contains exactly one global image.

This extension is not selected by training until its real GPU smoke test passes.
Each output independently covers every agent for that sample. The VLM never sees
multiple environments in one conversational context.
"""
import json
import time
from PIL import Image
import torch
from vlm4coop.teacher import LocalQwenTeacher, TeacherResult, joint_prompt, parse_probabilities, SYSTEM_PROMPT
from vlm4coop.thinking_budget import ThinkingBudget


class BatchedQwenTeacher(LocalQwenTeacher):
    def predict_many(self, images, n_agents, use_cache=True):
        if not images:
            return []
        images = [image.convert('RGB') for image in images]
        keys = [self.cache.key(image, n_agents) for image in images]
        results, missing, unique_positions = {}, [], {}
        for index, (image, key) in enumerate(zip(images, keys)):
            # Pure VLM: every image gets a real generation even for identical
            # states. Training: duplicate states can reuse the same frozen label.
            identity = key if use_cache else index
            if identity in unique_positions:
                continue
            unique_positions[identity] = index
            cached = self.cache.get(key) if use_cache else None
            if cached is not None:
                results[identity] = cached
            else:
                missing.append((identity, index))
        if not missing:
            return [results[key if use_cache else index] for index, key in enumerate(keys)]
        batch_size = self.config['batch_size']
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Invalid teacher batch size')
        closing_ids = self.processor.tokenizer.encode('</think>', add_special_tokens=False)
        if len(closing_ids) != 1:
            raise ValueError('Expected one closing-think token')
        for start in range(0, len(missing), batch_size):
            entries = missing[start:start + batch_size]
            batch_images = [images[index] for _, index in entries]
            messages = [[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': [
                {'type': 'image', 'image': image}, {'type': 'text', 'text': joint_prompt(n_agents)}]}]
                for image in batch_images]
            prompts = self.processor.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=self.config['enable_thinking'])
            if not all(prompt.endswith('<think>\n') for prompt in prompts):
                raise ValueError('Thinking template is not enabled')
            inputs = self.processor(text=prompts, images=batch_images, padding=True,
                                    return_tensors='pt').to(self.model.device)
            budget = ThinkingBudget(inputs['input_ids'].shape[-1], closing_ids[0],
                                    self.config['thinking_budget_tokens'])
            torch.cuda.synchronize()
            began = time.monotonic()
            from vlm4coop.structured_output import MatrixJsonOutput
            json_output = MatrixJsonOutput(self.format_tokenizer_data, inputs['input_ids'].shape[-1],
                closing_ids[0], n_agents, self.config['probability_decimals'])
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, do_sample=False,
                    max_new_tokens=self.config['max_new_tokens'], logits_processor=[budget, json_output])
            torch.cuda.synchronize()
            elapsed = time.monotonic() - began
            self.generate_calls += 1
            failures = []
            for row, (identity, index) in enumerate(entries):
                token_ids = outputs[row, inputs['input_ids'].shape[-1]:]
                eos = self.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                for end, value in enumerate(token_ids.tolist()):
                    if value in eos:
                        token_ids = token_ids[:end+1]
                        break
                raw = self.processor.decode(token_ids, skip_special_tokens=False)
                self.calls += 1
                event = dict(key=keys[index], n_agents=n_agents, image_count=1,
                    batch_size=len(entries), inference_batch=self.generate_calls,
                    batch_latency_seconds=elapsed,
                    amortized_state_latency_seconds=elapsed / len(entries),
                    generated_tokens=len(token_ids), raw=raw,
                    thinking_enabled=True, thinking_budget_tokens=self.config['thinking_budget_tokens'],
                    actual_thinking_tokens=(token_ids == closing_ids[0]).nonzero()[0].item()
                        if (token_ids == closing_ids[0]).any() else None)
                try:
                    if '</think>' not in raw:
                        raise ValueError('Thinking did not complete')
                    probs, repaired = parse_probabilities(raw, n_agents)
                    event.update(valid=True, repaired=repaired)
                    result = TeacherResult(probs, raw, repaired, elapsed, len(token_ids))
                    results[identity] = result
                    if use_cache:
                        self.cache.put(keys[index], result)
                except ValueError as exc:
                    event.update(valid=False, error=str(exc))
                    failures.append((index, str(exc)))
                with self.audit_path.open('a') as stream:
                    stream.write(json.dumps(event, allow_nan=False) + '\n')
            if failures:
                raise ValueError(f'Teacher output unrecoverable for {failures}; see {self.audit_path}')
        return [results[key if use_cache else index] for index, key in enumerate(keys)]

    def predict(self, image, n_agents, use_cache=True):
        return self.predict_many([image], n_agents, use_cache=use_cache)[0]

from pathlib import Path
p=Path('VLM4Coop/src/vlm4coop/api_teacher.py')
s=p.read_text().replace('import time\n', 'import time\nimport threading\n')
s=s.replace('        self.calls = 0\n', '        self.calls = 0\n        self._io_lock = threading.Lock()\n')
s=s.replace('        self.calls += 1\n', '        with self._io_lock:\n            self.calls += 1\n            event["call_index"] = self.calls\n')
s=s.replace("            with self.audit_path.open('a') as stream:\n                stream.write(json.dumps(self._safe(event), ensure_ascii=False, allow_nan=False) + '\\n')", "            with self._io_lock:\n                with self.audit_path.open('a') as stream:\n                    stream.write(json.dumps(self._safe(event), ensure_ascii=False, allow_nan=False) + '\\n')")
s=s.replace("        with self.audit_path.with_name('teacher_labels.jsonl').open('a') as stream:\n            stream.write(json.dumps(self._safe(accepted), allow_nan=False) + '\\n')", "        with self._io_lock:\n            with self.audit_path.with_name('teacher_labels.jsonl').open('a') as stream:\n                stream.write(json.dumps(self._safe(accepted), allow_nan=False) + '\\n')")
s += '''
    def predict_many(self, images, n_agents, use_cache=True):
        from vlm4coop.api_batch import predict_many
        return predict_many(self, images, n_agents, use_cache)
'''
p.write_text(s)
p=Path('VLM4Coop/src/vlm4coop/training.py')
s=p.read_text()
start=s.index('def annotate(')
end=s.index('\n\ndef evaluate(',start)
s=s[:start]+'''def annotate(runner, batch, teacher):
    """Freeze pre-action targets once; bound temporary PIL copies to 32 frames."""
    try:
        items = list(runner.frames.items())
        for start in range(0, len(items), 32):
            chunk = items[start:start + 32]
            if any(t >= batch.max_seq_length - 1 for (worker, t), rgb in chunk):
                raise ValueError("Captured terminal/padding frame")
            images = [Image.fromarray(rgb) for _, rgb in chunk]
            if hasattr(teacher, 'predict_many'):
                results = teacher.predict_many(images, runner.args.n_agents)
            else:
                results = [teacher.predict(image, runner.args.n_agents) for image in images]
            if len(results) != len(chunk):
                raise ValueError("Teacher returned a different number of state labels")
            for ((worker, t), _), result in zip(chunk, results):
                batch.update({"teacher_probs": torch.from_numpy(result.probs)[None, None],
                              "teacher_valid": torch.ones(1, 1, 1)},
                             bs=[worker], ts=t, mark_filled=False)
    finally:
        runner.frames.clear()
''' + s[end:]
p.write_text(s)
p=Path('VLM4Coop/src/vlm4coop/pure_vlm.py')
s=p.read_text()
start=s.index('        actions = []')
end=s.index('\n\n\ndef run(',start)
s=s[:start]+'''        # Native runner also selects terminal actions. Never query these states.
        actions = [[0] * self.n_agents for _ in bs]
        positions, images = [], []
        for index, worker in enumerate(bs):
            rgb = self.runner.frames.pop((worker, t_ep), None)
            if rgb is not None:
                positions.append(index)
                images.append(Image.fromarray(rgb))
        if hasattr(self.teacher, 'predict_many'):
            results = self.teacher.predict_many(images, self.n_agents, use_cache=False)
        else:
            results = [self.teacher.predict(image, self.n_agents, use_cache=False) for image in images]
        if len(results) != len(images):
            raise ValueError('Teacher did not return one label per active environment')
        for index, result in zip(positions, results):
            actions[index] = result.probs.argmax(-1).tolist()
        return torch.tensor(actions, dtype=torch.long)
''' + s[end:]
p.write_text(s)

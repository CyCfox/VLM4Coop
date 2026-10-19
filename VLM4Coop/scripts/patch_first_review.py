from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/teacher.py')
s = p.read_text()
s = s.replace('''        if any(type(v) not in (int, float) for v in values):
            raise ValueError("Non-numeric probability")
        p = np.asarray(values, dtype=np.float64)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Invalid probability with no safe maximum to preserve")
        if not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
            maximum = int(p.argmax())
            confidence = p[maximum]
''', '''        numeric = np.asarray([float(v) if type(v) in (int, float) else np.nan for v in values])
        if not np.isfinite(numeric).any():
            raise ValueError("No identifiable finite maximum")
        maximum = int(np.nanargmax(numeric))
        confidence = numeric[maximum]
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Maximum is not a legal probability; cannot preserve it")
        p = numeric.copy()
        valid = np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        if not valid or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0):
''')
p.write_text(s)
p = Path('VLM4Coop/src/vlm4coop/training.py')
s = p.read_text().replace('''            learner.train(batch, actual, episode)''', '''            # Match epymarl/src/run.py: discard trailing padding before reward
            # standardisation and native n-step return computation.
            batch = batch[:, :int(batch.max_t_filled())]
            learner.train(batch, actual, episode)''')
s = s.replace('''json.dumps({"train_env_steps": actual,
                                                                 "rank": list(best)})''', '''json.dumps(_json_safe({"train_env_steps": actual,
                                                                 "rank": list(best)}), allow_nan=False)''')
p.write_text(s)
p = Path('VLM4Coop/tests/test_core.py')
s = p.read_text() + '''\n\ndef test_parser_repairs_negative_entries_preserving_identifiable_maximum():
    p, repaired = parse_probabilities(raw([(0, [.8, -.1, .1, .1, .1, 0])]), 1)
    assert repaired
    assert p[0, 0] == pytest.approx(.8)
    assert np.allclose(p[0, 1:], .04)
'''
p.write_text(s)

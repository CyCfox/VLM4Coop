from pathlib import Path
p = Path('VLM4Coop/tests/test_integration.py')
s = p.read_text().replace('''            image1 = env.render().copy()
            for index''', '''            image1 = env.render().copy()
            obs_same, _ = env.reset(seed=7)
            assert np.array_equal(np.asarray(obs1), np.asarray(obs_same))
            assert np.array_equal(image1, env.render())
            for index''')
s += '''\n\ndef test_evaluation_rng_isolation():
    import random
    from vlm4coop.training import evaluation_rng
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    expected = random.random(), np.random.rand(), torch.rand(3)
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    with evaluation_rng(1000000):
        random.random()
        np.random.rand(17)
        torch.rand(8)
    actual = random.random(), np.random.rand(), torch.rand(3)
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
'''
p.write_text(s)

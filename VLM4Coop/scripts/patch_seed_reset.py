from pathlib import Path
p = Path('VLM4Coop/src/vlm4coop/environment.py')
s = p.read_text().replace('''    def reset(self, seed=None, options=None):
        obs, info = super().reset''', '''    def reset(self, seed=None, options=None):
        if seed is not None:
            # LBF 2.0 spawn_players otherwise rejects positions retained from
            # the previous episode. Clear public position fields to make a
            # fixed seeded scenario independent of the prior policy's end state.
            for player in self._env.unwrapped.players:
                player.position = None
        obs, info = super().reset''')
p.write_text(s)
p = Path('VLM4Coop/tests/test_integration.py')
s = p.read_text().replace('''            assert info['remaining_food'] == n
''', '''            assert info['remaining_food'] == n
            obs1, _ = env.reset(seed=7)
            image1 = env.render().copy()
            for index, player in enumerate(env._env.unwrapped.players):
                player.position = (index, index)
            obs2, _ = env.reset(seed=7)
            assert np.array_equal(np.asarray(obs1), np.asarray(obs2))
            assert np.array_equal(image1, env.render())
''')
p.write_text(s)

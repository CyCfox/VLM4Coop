"""Small real LBF/native-PPO checks, plus explicitly synthetic label plumbing."""
from copy import deepcopy
import json
import numpy as np
import torch
import pytest
from vlm4coop.config import load_config, make_args
from vlm4coop.environment import register_lbf, CoopGymmaWrapper
from vlm4coop.training import setup_runner, annotate, train
from vlm4coop.teacher import TeacherResult
from learners.ppo_learner import PPOLearner
from vlm4coop.learner import CoopPPOLearner


class Log:
    def log_stat(self, *args):
        pass


def test_map_size_registration_and_formal_teacher_guard(tmp_path):
    config = load_config()
    for size in (8, 10):
        variant = deepcopy(config)
        variant['environment']['field_size'] = [size, size]
        variant['environment']['sight'] = size
        key = register_lbf(3, variant['environment'])
        assert key == make_args(variant, 3, 0).env_args['key']
        env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None,
                              seed=0, common_reward=True, reward_scalarisation='sum')
        try:
            env.reset(seed=19)
            assert env._env.unwrapped.field.shape == (size, size)
            assert int(np.count_nonzero(env._env.unwrapped.field)) == 3
        finally:
            env.close()
    config['formal_authorized_methods'] = ['mappo']
    for status in ('mappo_authorized_teacher_pending', 'user_resolved_2026_10_17'):
        config['protocol_status'] = status
        for method in ('soft', 'hard'):
            with pytest.raises(ValueError, match='Unresolved formal protocol'):
                train(method, 3, 0, 'formal', tmp_path / method, config=config)


class SyntheticTeacher:
    """TEST DOUBLE ONLY. Never a formal experiment baseline or teacher."""
    calls = 0
    def predict(self, image, n_agents):
        self.calls += 1
        p = np.full((n_agents, 6), .1, dtype=np.float32)
        p[:, 5] = .5
        return TeacherResult(p, 'SYNTHETIC UNIT TEST LABEL', False)


def test_lbf_shapes_and_real_timeout():
    config = load_config()
    for n in (3, 4, 5):
        key = register_lbf(n, config['environment'])
        env = CoopGymmaWrapper(key=key, time_limit=100, pretrained_wrapper=None,
                              seed=0, common_reward=True, reward_scalarisation='sum')
        try:
            obs, _ = env.reset(seed=19)
            assert np.asarray(obs).shape == (n, 6*n)
            image = env.render()
            rows, cols = config['environment']['field_size']
            assert image.shape == (rows * 56 + 28, cols * 56 + 28, 3) and image.dtype == np.uint8
            for t in range(100):
                _, _, done, trunc, info = env.step([0]*n)
                assert bool(done or trunc) == (t == 99)
            assert not info['success'] and info['episode_limit']
            assert info['remaining_food'] == n
            obs1, _ = env.reset(seed=7)
            image1 = env.render().copy()
            obs_same, _ = env.reset(seed=7)
            assert np.array_equal(np.asarray(obs1), np.asarray(obs_same))
            assert np.array_equal(image1, env.render())
            for index, player in enumerate(env._env.unwrapped.players):
                player.position = (index, index)
            obs2, _ = env.reset(seed=7)
            assert np.array_equal(np.asarray(obs1), np.asarray(obs2))
            assert np.array_equal(image1, env.render())
        finally:
            env.close()


def test_native_zero_kd_parameter_parity():
    config = load_config()
    args = make_args(config, 3, 0, 'smoke')
    args.kd_lambda_current, args.kd_mode = 0., 'soft'
    torch.manual_seed(13)
    runner, mac, scheme = setup_runner(args, Log(), config)
    try:
        with torch.no_grad():
            batch = runner.run(test_mode=False)
        a = PPOLearner(deepcopy(mac), scheme, Log(), deepcopy(args))
        torch.manual_seed(42)
        b = CoopPPOLearner(deepcopy(mac), scheme, Log(), deepcopy(args))
        # Align critic initialization as well as copied actor weights.
        b.critic.load_state_dict(a.critic.state_dict())
        b.target_critic.load_state_dict(a.target_critic.state_dict())
        a.train(batch, runner.t_env, 0)
        b.train(batch, runner.t_env, 0)
        for x, y in zip(a.mac.parameters(), b.mac.parameters()):
            assert torch.allclose(x, y, atol=1e-7, rtol=1e-6)
        for x, y in zip(a.critic.parameters(), b.critic.parameters()):
            assert torch.allclose(x, y, atol=1e-7, rtol=1e-6)
        assert runner.t_env == sum(item['length'] for item in runner.last_episode_records)
    finally:
        runner.close_env()


@pytest.mark.parametrize('mode',['soft','hard'])
def test_format_failure_masks_only_distillation_and_keeps_native_ppo_updates(tmp_path,mode):
    config=load_config()
    args=make_args(config,3,0,'smoke')
    args.kd_lambda_current,args.kd_mode=10.,mode
    torch.manual_seed(13)
    runner,mac,scheme=setup_runner(args,Log(),config)
    class FailedTeacher:
        config={'concurrency':1000}
        cache=None
        audit_path=tmp_path/'calls.jsonl'
        def predict_many(self,images,n_agents):
            # Synthetic post-retry failure; actual 3-request bound is tested
            # separately against mocked HTTP in test_response_protocol.py.
            failure=TeacherResult(None,'SYNTHETIC FORMAT FAILURE',False,
                                  source='format_failure',format_attempts=3)
            return [failure]*len(images)
    try:
        runner.capture_frames=True
        with torch.no_grad(): batch=runner.run(test_mode=False)
        original={name:batch[name].clone() for name in ('filled','actions','reward','terminated')}
        annotate(runner,batch,FailedTeacher())
        assert not batch['teacher_valid'].any() and not batch['teacher_probs'].any()
        for name,value in original.items(): assert torch.equal(batch[name],value)
        audit=[json.loads(row) for row in (tmp_path/'teacher_targets.jsonl').read_text().splitlines()]
        assert len(audit)==runner.t_env
        assert all(row['teacher_valid']==0 and row['probabilities'] is None and row['source']=='kd_masked' for row in audit)
        a=PPOLearner(deepcopy(mac),scheme,Log(),deepcopy(args))
        b=CoopPPOLearner(deepcopy(mac),scheme,Log(),deepcopy(args))
        b.critic.load_state_dict(a.critic.state_dict())
        b.target_critic.load_state_dict(a.target_critic.state_dict())
        initial=[p.detach().clone() for p in mac.parameters()]
        a.train(batch,runner.t_env,0); b.train(batch,runner.t_env,0)
        assert any(not torch.equal(x,y) for x,y in zip(initial,b.mac.parameters()))
        for x,y in zip(a.mac.parameters(),b.mac.parameters()):
            assert torch.allclose(x,y,atol=1e-7,rtol=1e-6)
        for x,y in zip(a.critic.parameters(),b.critic.parameters()):
            assert torch.allclose(x,y,atol=1e-7,rtol=1e-6)
    finally: runner.close_env()


def test_frames_and_teacher_labels_once_per_real_state():
    config = load_config()
    args = make_args(config, 3, 0, 'smoke')
    runner, mac, scheme = setup_runner(args, Log(), config)
    teacher = SyntheticTeacher()
    try:
        runner.capture_frames = True
        with torch.no_grad():
            batch = runner.run(test_mode=False)
        count = len(runner.frames)
        assert count == runner.t_env
        annotate(runner, batch, teacher)
        assert not runner.frames and teacher.calls == count
        assert int(batch['teacher_valid'].sum()) == count
        args.kd_lambda_current, args.kd_mode = .1, 'soft'
        learner = CoopPPOLearner(mac, scheme, Log(), args)
        learner.train(batch, runner.t_env, 0)
        assert teacher.calls == count  # all four PPO epochs reused cached targets
        with torch.no_grad():
            runner.capture_frames = False
            runner.run(test_mode=False)
        assert not runner.frames
    finally:
        runner.close_env()


def test_evaluation_rng_isolation():
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


def test_full_fixed_evaluation_and_pure_teacher_control(tmp_path):
    from vlm4coop.pure_vlm import TeacherMAC
    from vlm4coop.training import evaluate

    class FixedActionTeacher(SyntheticTeacher):
        def predict(self, image, n_agents, use_cache=True):
            assert use_cache  # Legacy cache-enabled plumbing; formal Pure VLM explicitly passes False.
            return super().predict(image, n_agents)

    config = load_config()
    args = make_args(config, 3, 0)
    teacher = FixedActionTeacher()
    mac = TeacherMAC(teacher, 3)
    runner, _, _ = setup_runner(args, Log(), config, mac)
    try:
        runner.capture_frames, mac.runner = True, runner
        result = evaluate(runner, config, tmp_path, 1000, 1000, 0)
        records = [json.loads(line) for line in (tmp_path / 'eval_episodes.jsonl').read_text().splitlines()]
        assert result['n_eval'] == 100 and len(records) == 100
        assert [record['eval_seed'] for record in records] == list(range(1000000, 1000100))
        assert teacher.calls == sum(record['length'] for record in records)
        assert runner.t_env == 0  # evaluation never advances training count
        assert not runner.frames
    finally:
        runner.close_env()


def test_eval_failure_preserves_completed_ppo_progress(monkeypatch, tmp_path):
    from vlm4coop import training
    config = load_config()
    monkeypatch.setattr(training, 'PROJECT', tmp_path)
    def unavailable_eval(*args, **kwargs):
        raise RuntimeError('intentional evaluation failure')
    monkeypatch.setattr(training, 'evaluate', unavailable_eval)
    output = tmp_path / 'failed_eval'
    with pytest.raises(RuntimeError, match='intentional evaluation failure'):
        train('mappo', 3, 0, 'smoke', output, config=config)
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert status['train_env_steps'] == status['sampled_env_steps'] == 1000


def test_failed_second_teacher_batch_persists_current_counters(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from vlm4coop import training
    from vlm4coop.api_teacher import ApiRequestError
    monkeypatch.setattr(training,'PROJECT',tmp_path)
    class FailingTeacher:
        config={'concurrency':1000}
        calls=max_active_requests=format_failures=format_retries=0
        thinking_control_retries=thinking_control_failures=failed_decisions=0
        cache=SimpleNamespace(hits=0,misses=0,writes=0)
        batches=0
        def predict_many(self,images,n_agents):
            self.batches+=1
            if self.batches==1:
                self.cache.hits=len(images)
                return [TeacherResult(np.full((n_agents,6),1/6,dtype=np.float32),
                    'SYNTHETIC CACHED LABEL',False,cache_hit=True) for _ in images]
            self.calls,self.cache.misses,self.cache.writes=17,5,3
            self.max_active_requests=5
            self.thinking_control_retries,self.thinking_control_failures=7,8
            raise ApiRequestError('synthetic OFF control exhaustion')
    output=tmp_path/'teacher_failure'
    teacher=FailingTeacher()
    with pytest.raises(ApiRequestError,match='synthetic OFF'):
        train('hard',3,0,'smoke',output,config=load_config(),teacher=teacher)
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='failed' and status['phase']=='failed'
    assert status['train_env_steps']==1000 and status['sampled_env_steps']==2000 and status['episodes']==10
    assert status['teacher_calls']==17 and status['cache_misses']==5 and status['cache_writes']==3
    assert status['peak_active_http_requests']==5
    assert status['thinking_control_retries']==7 and status['rejected_thinking_responses']==8
    assert len((output/'evaluations.jsonl').read_text().splitlines())==1
    assert status['wall_seconds']>0

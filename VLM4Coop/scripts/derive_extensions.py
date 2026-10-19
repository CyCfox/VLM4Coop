"""One-time source extraction. Run only in Docker; generated files are committed artifacts.

Copy only the native PPO train method and native parallel runner, then add the
minimal teacher/data hooks. All other learner methods remain inherited.
"""
from pathlib import Path

root = Path('/workspace/my_project')
out = root / 'VLM4Coop/src/vlm4coop'
source = (root / 'epymarl/src/learners/ppo_learner.py').read_text()
method = source[source.index('    def train('):source.index('    def train_critic_sequential(')]
method = method.replace('            # Optimise agents\n', '''            # VLM4Coop: use cached labels; never query a teacher in PPO epochs.
            weight = self.args.kd_lambda_current
            kd = distillation_loss(pi, batch["teacher_probs"][:, :-1],
                                   mask * batch["teacher_valid"][:, :-1],
                                   self.args.kd_mode) if weight > 0 else pi.sum() * 0.0
            pg_loss = pg_loss + weight * kd
            self.logger.log_stat("kd_loss", kd.item(), t_env)
            self.logger.log_stat("kd_weight", weight, t_env)

            # Optimise agents
''')
(out / 'learner.py').write_text('''# PPO train method adapted from EPyMARL (Apache-2.0), commit
# cbc38c09588064eab978501d0f12c2cf58fa7fc2. Added cached KD only.
import torch as th
from components.episode_buffer import EpisodeBatch
from learners.ppo_learner import PPOLearner
from vlm4coop.distillation import distillation_loss


class CoopPPOLearner(PPOLearner):
''' + method)
source = (root / 'epymarl/src/runners/parallel_runner.py').read_text()
source = '# Adapted from EPyMARL ParallelRunner (Apache-2.0), commit cbc38c09588064eab978501d0f12c2cf58fa7fc2.\n# Changes: action-before RGB, fixed evaluation resets, episode records, cleanup.\n' + source
source = source.replace('class ParallelRunner:', 'class CoopParallelRunner:')
source = source.replace('        self.batch = self.new_batch()\n', '        self.batch = self.new_batch()\n        self.frames = {}  # local to THIS rollout; never reused across batches\n')
source = source.replace('        for parent_conn in self.parent_conns:\n            parent_conn.send(("reset", None))', '        seeds = getattr(self, "next_episode_seeds", [None] * self.batch_size)\n        for parent_conn, seed in zip(self.parent_conns, seeds):\n            parent_conn.send(("reset", seed))')
source = source.replace('        while True:\n            # Pass', '''        while True:
            if getattr(self, "capture_frames", False):
                active = [i for i in range(self.batch_size) if not terminated[i]]
                for i in active:
                    self.parent_conns[i].send(("rgb", None))
                for i in active:
                    self.frames[(i, self.t)] = self.parent_conns[i].recv()
            # Pass''')
source = source.replace('        final_env_infos = []', '        self.last_episode_records = [None] * self.batch_size\n        final_env_infos = []')
source = source.replace('                        final_env_infos.append(data["info"])', '''                        final_env_infos.append(data["info"])
                        self.last_episode_records[idx] = dict(
                            worker=idx, length=episode_lengths[idx],
                            team_return=float(episode_returns[idx]),
                            success=bool(data["info"]["success"]),
                            initial_food_count=data["info"]["initial_food_count"],
                            remaining_food=data["info"]["remaining_food"],
                            episode_limit=bool(data["info"].get("episode_limit", False)))''')
source = source.replace('            env.reset()\n', '            env.reset(seed=data)\n')
source = source.replace("elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:", "elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:")
source = source.replace('        elif cmd == "render":', '        elif cmd == "rgb":\n            remote.send(env.render())\n        elif cmd == "render":')
source = source.replace('            parent_conn.send(("close", None))\n', '''            parent_conn.send(("close", None))
        for process in self.ps:
            process.join(timeout=10)
        for parent_conn in self.parent_conns:
            parent_conn.close()
''')
(out / 'runner.py').write_text(source)

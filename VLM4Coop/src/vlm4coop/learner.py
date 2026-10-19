# PPO train method adapted from EPyMARL (Apache-2.0), commit
# cbc38c09588064eab978501d0f12c2cf58fa7fc2. Added cached KD only.
import torch as th
from components.episode_buffer import EpisodeBatch
from learners.ppo_learner import PPOLearner
from vlm4coop.distillation import distillation_loss


class CoopPPOLearner(PPOLearner):
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        for k in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            pi = mac_out
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()
            # Calculate policy grad with mask

            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = (
                -(
                    (th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask
                ).sum()
                / mask.sum()
            )

            # Format failures disable KD only. The native PPO/critic masks
            # above retain these transitions for all four update epochs.
            weight = self.args.kd_lambda_current
            kd = distillation_loss(pi, batch["teacher_probs"][:, :-1],
                                   mask * batch["teacher_valid"][:, :-1],
                                   self.args.kd_mode) if weight > 0 else pi.sum() * 0.0
            pg_loss = pg_loss + weight * kd
            self.logger.log_stat("kd_loss", kd.item(), t_env)
            self.logger.log_stat("kd_weight", weight, t_env)

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env


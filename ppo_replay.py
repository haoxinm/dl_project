import os
import time
from collections import deque

import torch
import torch.nn as nn
from habitat import Config, logger
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    linear_decay,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from torch.optim.lr_scheduler import LambdaLR
from replay_buffer import RolloutReplayBuffer
from efficientnet_policy import PointNavEfficientNetPolicy


@baseline_registry.register_trainer(name="ppo_replay")
class PPOReplayTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.memory = RolloutReplayBuffer(config.REPLAY_MEMORY_SIZE)

    def insert_memory(self, memories):
        for rollout, reward, count in memories:
            self.memory.insert(rollout, reward, count)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = PointNavEfficientNetPolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.PPO.rnn_type,
            num_recurrent_layers=self.config.RL.PPO.num_recurrent_layers,
            backbone=self.config.RL.PPO.backbone,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs="rgb"
                                    in self.envs.observation_spaces[0].spaces,
            pretrained=self.config.RL.PPO.PRETRAINED,
            finetune=self.config.RL.PPO.FINETUNE,
        )
        self.actor_critic.to(self.device)

        if (
            self.config.RL.PPO.pretrained_encoder
            or self.config.RL.PPO.pretrained_actor
        ):
            pretrained_state = torch.load(
                self.config.RL.PPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.PPO.pretrained_actor:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.PPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.PPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.PPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def _update_agent_memory(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        # rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def replay(self, num_updates, ppo_cfg, lr_scheduler, t_start, pth_time, writer,
               count_steps, count_checkpoints):
        print(".....start memory replay for {} updates.....".format(num_updates))
        env_time = 0
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        memories = self.memory.recall(num_updates)
        for update in range(num_updates):
            rollouts, episode_rewards, episode_counts = memories[update]
            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                    update, self.config.NUM_UPDATES
                )
            (
                delta_pth_time,
                value_loss,
                action_loss,
                dist_entropy,
            ) = self._update_agent_memory(ppo_cfg, rollouts)
            pth_time += delta_pth_time

            window_episode_reward.append(episode_rewards.clone())
            window_episode_counts.append(episode_counts.clone())

            losses = [value_loss, action_loss]
            stats = zip(
                ["count", "reward"],
                [window_episode_counts, window_episode_reward],
            )
            deltas = {
                k: (
                    (v[-1] - v[0]).sum().item()
                    if len(v) > 1
                    else v[0].sum().item()
                )
                for k, v in stats
            }
            deltas["count"] = max(deltas["count"], 1.0)

            writer.add_scalar(
                "reward", deltas["reward"] / deltas["count"], count_steps
            )

            writer.add_scalars(
                "losses",
                {k: l for l, k in zip(losses, ["value", "policy"])},
                count_steps,
            )

            # log stats
            if update > 0 and update % self.config.LOG_INTERVAL == 0:
                logger.info(
                    "update: {}\tfps: {:.3f}\t".format(
                        update, count_steps / (time.time() - t_start)
                    )
                )

                logger.info(
                    "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                    "frames: {}".format(
                        update, env_time, pth_time, count_steps
                    )
                )

                window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                ).sum()
                window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                ).sum()

                if window_counts > 0:
                    logger.info(
                        "Average window size {} reward: {:3f}".format(
                            len(window_episode_reward),
                            (window_rewards / window_counts).item(),
                        )
                    )
                else:
                    logger.info("No episodes finish in current window")

            # checkpoint model
            if update % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                )
                count_checkpoints += 1

            count_steps += 1

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:

            # train with human demonstrations
            self.replay(self.config.NUM_DEMO_UPDATES, ppo_cfg, lr_scheduler, t_start,
                        pth_time, writer, count_steps, count_checkpoints)

            # train with agent experiences
            for update in range(self.config.NUM_UPDATES):
                if update != 0 and update % self.config.REPLAY_INTERVAL == 0:
                    self.replay(self.config.NUM_REPLAY_UPDATES, ppo_cfg, lr_scheduler, t_start,
                                pth_time, writer, count_steps, count_checkpoints)

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        episode_rewards,
                        episode_counts,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                self.insert_memory([rollouts, episode_rewards, episode_counts])

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                window_episode_reward.append(episode_rewards.clone())
                window_episode_counts.append(episode_counts.clone())

                losses = [value_loss, action_loss]
                stats = zip(
                    ["count", "reward"],
                    [window_episode_counts, window_episode_reward],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

            self.envs.close()

import os
import time
from collections import deque, defaultdict
from apex import amp
import torch
import torch.nn as nn
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    linear_decay,
)
from habitat_baselines.rl.ppo import PPO, PointNavBaselinePolicy
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from efficientnet_policy import PointNavEfficientNetPolicy
from replay_buffer import RolloutReplayBuffer
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.sims.habitat_simulator.actions import HabitatSimActions

cv2 = try_cv2_import()


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


@baseline_registry.register_trainer(name="ppo_replay")
class PPOReplayTrainer(PPOTrainer):
    METRICS_BLACKLIST = {}  # "top_down_map", "collisions.is_collision"

    def __init__(self, config=None):
        super().__init__(config)
        self.memory = RolloutReplayBuffer(config.REPLAY_MEMORY_SIZE)

    def insert_memory(self, rollout):
        # for rollout, stat in memories:
        self.memory.insert(rollout)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        # self.actor_critic = PointNavEfficientNetPolicy(
        #     observation_space=self.envs.observation_spaces[0],
        #     action_space=self.envs.action_spaces[0],
        #     hidden_size=ppo_cfg.hidden_size,
        #     rnn_type=self.config.RL.PPO.rnn_type,
        #     num_recurrent_layers=self.config.RL.PPO.num_recurrent_layers,
        #     backbone=self.config.RL.PPO.backbone,
        #     goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
        #     normalize_visual_inputs="rgb"
        #                             in self.envs.observation_spaces[0].spaces,
        #     pretrained=self.config.RL.PPO.PRETRAINED,
        #     finetune=self.config.RL.PPO.FINETUNE,
        # )
        self.actor_critic = PointNavBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
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
                    k[len("actor_critic."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.PPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.PPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.PPO.reset_critic:
            self.actor_critic.critic.reset()
            # nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            # nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = PPODistributed(
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

    def _collect_rollout_step(
            self, rollouts, current_episode_reward, running_episode_stats, update=0,
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # rollouts.to(self.device)
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        best_actions = torch.tensor(self.envs.call(["get_best_action"] * self.envs.num_envs),
                                    device=actions.device)
        # tf = False
        # if np.random.random() < 1. - (update / self.config.NUM_TF_UPDATES) * \
        #         (update > 2 * self.config.NUM_REPLAY_UPDATES):
        #     # tf = True
        #     actions = best_actions
        # else:
        action_list = [HabitatSimActions.MOVE_FORWARD, HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]
        punishment = torch.zeros(actions.shape)
        if update < self.config.NUM_TAMER_UPDATES // 2:
            for i in range(self.envs.num_envs):
                if step_observation[self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID][i][0] > \
                        self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE and \
                        actions[i] == HabitatSimActions.STOP:
                    if np.random.random() < 0.5 * (1 - update < self.config.NUM_TAMER_UPDATES // 2):
                        actions[i] = torch.tensor(np.random.choice(action_list), dtype=actions.dtype, device=actions.device)
                    else:
                        punishment[i] = torch.tensor(self.config.RL.EARLY_STOP_PUNISHMENT)
        tamer_rewards = (2 * (actions == best_actions) - 1) * self.config.RL.TAMER_REWARD
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        top_down_map = []
        for i in range(self.envs.num_envs):
            top_down_map.append(np.transpose(draw_top_down_map(
                infos[i], observations[i]["heading"][0], observations[i]['depth'].shape[0]
            ), [2, 0, 1]))
            if infos[i]['collisions']['is_collision']:
                rewards[i] += self.config.RL.COLLISION_REWARD

        env_time += time.time() - t_step_env
        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)
        # if not tf:
        rewards += tamer_rewards.to(rewards.device) * (update < self.config.NUM_TAMER_UPDATES)
        rewards += punishment.to(rewards.dtype).to(rewards.device)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs, top_down_map

    def _update_agent_memory(self, ppo_cfg, rollouts):
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

        # return (
        #     time.time() - t_update_model,
        #     value_loss,
        #     action_loss,
        #     dist_entropy,
        # )

    def replay(self, num_updates, ppo_cfg, lr_scheduler):
        print(".....start memory replay for {} updates.....".format(num_updates))
        rollouts_memories = self.memory.recall(num_updates)
        for update in range(num_updates):
            rollouts_memory = rollouts_memories[update]
            rollouts_memory.to(self.env_device)
            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()
            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                    update, self.config.NUM_UPDATES
                )
            # torch.cuda.empty_cache()
            # (
            #     delta_pth_time,
            #     value_loss,
            #     action_loss,
            #     dist_entropy,
            # ) = \
            self._update_agent_memory(ppo_cfg, rollouts_memory)
            rollouts_memory.to('cpu')

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
        self.env_device = (
            torch.device("cuda", self.config.SIMULATOR_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        if self.config.FP16:
            self.half()
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
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        # rollouts.to('cpu')
        rollouts.to(self.env_device)

        observations = self.envs.reset()
        batch = batch_obs(observations)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

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
            self.replay(self.config.NUM_DEMO_UPDATES, ppo_cfg, lr_scheduler)

            # train with agent experiences
            for update in range(self.config.NUM_UPDATES):
                if update != 0 and self.config.REPLAY_INTERVAL > 0 and update % self.config.REPLAY_INTERVAL == 0 and len(
                        self.memory) > self.config.NUM_REPLAY_UPDATES:
                    self.replay(self.config.NUM_REPLAY_UPDATES, ppo_cfg, lr_scheduler)

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
                        top_down_map,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats, update)
                    for i in range(len(top_down_map)):
                        writer.add_image('top down map-env{}'.format(i), top_down_map[i],
                                         step + update * ppo_cfg.num_steps)
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                if self.config.REPLAY_INTERVAL > 0:
                    self.insert_memory(rollouts)

                # torch.cuda.empty_cache()
                # rollouts.to(self.device)
                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [value_loss, action_loss]
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

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

            self.envs.close()

    def half(self):
        # self.actor_critic.half()
        # self.agent.half()
        self.agent, self.agent.optimizer \
            = amp.initialize(self.agent, self.agent.optimizer,
                             opt_level="O1")


class PPODistributed(PPO):
    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # torch.cuda.empty_cache()
                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch.to(action_log_probs.device)
                )
                adv_targ = adv_targ.to(ratio.device)
                surr1 = ratio * adv_targ
                surr2 = (
                        torch.clamp(
                            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                        )
                        * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)
                    value_pred_clipped = value_preds_batch + (
                            values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                            0.5
                            * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss = (
                        value_loss * self.value_loss_coef
                        + action_loss
                        - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

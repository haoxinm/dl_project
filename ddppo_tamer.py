import torch
import contextlib
import os
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch.distributed as distrib
import torch.nn as nn
from apex import amp
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from habitat import Config, logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs, linear_decay
from habitat_baselines.rl.ddppo import DDPPOTrainer
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from torch.optim.lr_scheduler import LambdaLR

from efficientnet_policy import PointNavEfficientNetPolicy, PointNavEfficientNetGPSPolicy
from ppo_replay import draw_top_down_map


@baseline_registry.register_trainer(name="ddppo_tamer")
class DDPPOTamerTrainer(DDPPOTrainer):
    METRICS_BLACKLIST = {}  # "top_down_map", "collisions.is_collision"

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        # self.actor_critic = PointNavResNetPolicy(
        #     observation_space=self.envs.observation_spaces[0],
        #     action_space=self.envs.action_spaces[0],
        #     hidden_size=ppo_cfg.hidden_size,
        #     rnn_type=self.config.RL.DDPPO.rnn_type,
        #     num_recurrent_layers=self.config.RL.DDPPO.num_recurrent_layers,
        #     backbone=self.config.RL.DDPPO.backbone,
        #     goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
        #     normalize_visual_inputs="rgb"
        #     in self.envs.observation_spaces[0].spaces,
        # )
        # self.actor_critic = PointNavBaselinePolicy(
        #     observation_space=self.envs.observation_spaces[0],
        #     action_space=self.envs.action_spaces[0],
        #     hidden_size=ppo_cfg.hidden_size,
        #     goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
        # )
        self.actor_critic = PointNavEfficientNetGPSPolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.DDPPO.rnn_type,
            num_recurrent_layers=self.config.RL.DDPPO.num_recurrent_layers,
            backbone=self.config.RL.DDPPO.backbone,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs="rgb"
                                    in self.envs.observation_spaces[0].spaces,
            pretrained=self.config.RL.DDPPO.pretrained_encoder,
            finetune=self.config.RL.DDPPO.finetune,
        )
        self.actor_critic.to(self.device)

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPOGPS(
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
        # gps = None
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            prev_observation = {
                k: v[rollouts.step] for k, v in rollouts.prev_observations.items()
            }
            prev_gps = step_observation['pointgoal']
            for i in range(self.envs.num_envs):
                if self.gps[i][0] >= 0.:
                    prev_gps[i] = self.gps[i]
            self.gps = step_observation['pointgoal_with_gps_compass']
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                gps_pred,
            ) = self.actor_critic.act(
                step_observation,
                prev_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                gps=self.gps,
                prev_gps=prev_gps,
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        best_actions = torch.tensor(self.envs.call(["get_best_action"] * self.envs.num_envs),
                                    device=actions.device)
        # tf = False
        # if self.config.NUM_TF_UPDATES !=0 and \
        #         np.random.random() < 1. - (update / self.config.NUM_TF_UPDATES) * \
        #         (update > 2 * self.config.NUM_REPLAY_UPDATES):
        #     # tf = True
        #     actions = best_actions
        # else:
        # action_list = [HabitatSimActions.MOVE_FORWARD, HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]
        tamer_rewards = (actions == best_actions) * (self.config.RL.TAMER_REWARD) - .075
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        punish = 0.
        punishment = torch.zeros(actions.shape)
        top_down_map = []
        for i in range(self.envs.num_envs):
            if dones[i]:
                self.gps[i][0] = -1.
            top_down_map.append(np.transpose(draw_top_down_map(
                infos[i], observations[i]["heading"][0], observations[i]['depth'].shape[0]
            ), [2, 0, 1]))
            # '''
            if (actions[i] == HabitatSimActions.STOP and
                infos[i]['distance_to_goal'] >
                self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE) \
                    or (infos[i]['distance_to_goal'] <
                     self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE and
                     actions[i] != HabitatSimActions.STOP):
                # if np.random.random() < 0.5 * (1 - update < self.config.NUM_TAMER_UPDATES // 2):
                #     actions[i] = torch.tensor(np.random.choice(action_list), dtype=actions.dtype, device=actions.device)
                # else:
                # punishment[i] = torch.tensor(self.config.RL.EARLY_STOP_PUNISHMENT)
                punish = max(self.config.RL.EARLY_STOP_PUNISHMENT, infos[i]['distance_to_goal'])
                punishment[i] = -punish
            # if update < self.config.NUM_TAMER_UPDATES // 2:
            #     if infos[i]['collisions']['is_collision']:
            #         rewards[i] += self.config.RL.COLLISION_REWARD
            # '''

        env_time += time.time() - t_step_env
        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)
        # if not tf:
        rewards += tamer_rewards.to(rewards.device)  #* (1. - update / self.config.NUM_TAMER_UPDATES)
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
            step_observation,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
            gps_pred,
            prev_gps,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs, top_down_map

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            prev_observation = {
                k: v[rollouts.step] for k, v in rollouts.prev_observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                prev_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                gps=last_observation['pointgoal_with_gps_compass'],
                prev_gps=rollouts.prev_gps[rollouts.step - 1],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy, gps_loss = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            gps_loss,
        )

    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        random.seed(self.config.TASK_CONFIG.SEED + self.world_rank)
        np.random.seed(self.config.TASK_CONFIG.SEED + self.world_rank)

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        self.config.freeze()

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        self.gps = torch.zeros((self.envs.num_envs, 2)) - 1.

        ppo_cfg = self.config.RL.PPO
        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        if self.config.FP16:
            self.half()
        self.agent.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        obs_space = self.envs.observation_spaces[0]
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = SpaceDict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts = GPSRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        )

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
                for step in range(ppo_cfg.num_steps):

                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                        top_down_map,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats, update
                    )
                    if writer is not None:
                        for i in range(len(top_down_map)):
                            writer.add_image('top down map-env{}'.format(i), top_down_map[i],
                                             step + update * ppo_cfg.num_steps)
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()
                if self._static_encoder:
                    self._encoder.eval()

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                    gps_loss,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                stats_ordering = list(sorted(running_episode_stats.keys()))
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [value_loss, action_loss, gps_loss, count_steps_delta],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[-1].item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                        stats[2].item() / self.world_size,
                    ]
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
                        "reward",
                        deltas["reward"] / deltas["count"],
                        count_steps,
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

                    writer.add_scalars(
                        "losses",
                        {k: l for l, k in zip(losses, ["value", "policy", "gps"])},
                        count_steps,
                    )

                    # log stats
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update,
                                count_steps
                                / ((time.time() - t_start) + prev_time),
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
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        count_checkpoints += 1

            self.envs.close()

    def half(self):
        # self.actor_critic.half()
        # self.agent.half()
        self.agent, self.agent.optimizer \
            = amp.initialize(self.agent, self.agent.optimizer,
                             opt_level="O1")


class DDPPOGPS(DDPPO):
    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        gps_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    prev_obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    gps_pred_batch,
                    prev_gps_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    prev_obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                    gps=obs_batch['pointgoal_with_gps_compass'],
                    prev_gps=prev_gps_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
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

                gps = obs_batch['pointgoal_with_gps_compass']
                gps = torch.stack(
                    [
                        gps[..., 0],
                        torch.cos(-gps[..., 1]),
                        torch.sin(-gps[..., 1]),
                    ],
                    -1,
                )
                gps_pred_batch = torch.stack(
                    [
                        gps_pred_batch[..., 0],
                        torch.cos(-gps_pred_batch[..., 1]),
                        torch.sin(-gps_pred_batch[..., 1]),
                    ],
                    -1,
                )
                gps_loss = 0.5 * (gps - gps_pred_batch).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss + gps_loss
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
                gps_loss_epoch += gps_loss.item()


        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        gps_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, gps_loss_epoch


class GPSRolloutStorage(RolloutStorage):
    def __init__(self, num_steps, num_envs, observation_space, action_space, recurrent_hidden_state_size,
                 num_recurrent_layers=1):
        super().__init__(num_steps, num_envs, observation_space, action_space, recurrent_hidden_state_size,
                         num_recurrent_layers)
        self.prev_observations = {}

        for sensor in observation_space.spaces:
            self.prev_observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )
        self.gps = torch.zeros(num_steps, num_envs, 2)
        self.prev_gps = torch.zeros(num_steps, num_envs, 2)

    def to(self, device):
        super(GPSRolloutStorage, self).to(device)
        self.gps = self.gps.to(device)
        self.prev_gps = self.prev_gps.to(device)

    def insert(
            self,
            observations,
            prev_observations,
            recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            gps_pred,
            prev_gps,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
            self.prev_observations[sensor][self.step + 1].copy_(
                prev_observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.gps[self.step].copy_(gps_pred)
        self.prev_gps[self.step].copy_(prev_gps)
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)
            prev_observations = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            gps_pred_batch = []
            prev_gps_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )
                    prev_observations[sensor].append(
                        self.prev_observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                gps_pred_batch.append(self.gps[: self.step, ind])
                prev_gps_batch.append(self.prev_gps[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )

                adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )
                prev_observations[sensor] = torch.stack(
                    prev_observations[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            gps_pred_batch = torch.stack(gps_pred_batch, 1)
            prev_gps_batch = torch.stack(prev_gps_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )
                prev_observations[sensor] = self._flatten_helper(
                    T, N, prev_observations[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            gps_pred_batch = self._flatten_helper(T, N, gps_pred_batch)
            prev_gps_batch = self._flatten_helper(T, N, prev_gps_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                prev_observations,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                gps_pred_batch,
                prev_gps_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

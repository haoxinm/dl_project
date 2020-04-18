import os
import time
from collections import defaultdict
from typing import Dict, Optional

import habitat
import numpy as np
import torch
from apex import amp
from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import batch_obs, linear_decay
from torch.optim.lr_scheduler import LambdaLR

from simple_odom import SimpleOdomNet


class RandomAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid, num_envs):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid
        self.num_envs = num_envs

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations[self.goal_sensor_uuid][:, 0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations):
        actions = []
        reached = self.is_goal_reached(observations)
        for i in range(self.num_envs):
            if reached[i]:
                action = HabitatSimActions.STOP
            else:
                action = np.random.choice(
                    [
                        HabitatSimActions.MOVE_FORWARD,
                        HabitatSimActions.TURN_LEFT,
                        HabitatSimActions.TURN_RIGHT,
                    ]
                )
            actions.append(action)
        return torch.tensor(actions, dtype=torch.long)


class OdomRolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )


        self.goal_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.goals = torch.zeros(num_steps, num_envs, 2)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.goal_log_probs = self.goal_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.goals = self.goals.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        goals,
        goal_log_probs,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.goal_log_probs[self.step].copy_(goal_log_probs)
        self.goals[self.step].copy_(goals)
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.step = 0

    def recurrent_generator(self, num_processes, num_mini_batch):
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            goals_batch = []
            masks_batch = []
            old_goal_log_probs_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                goals_batch.append(self.goals[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_goal_log_probs_batch.append(
                    self.goal_log_probs[: self.step, ind]
                )

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            goals_batch = torch.stack(goals_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_goal_log_probs_batch = torch.stack(
                old_goal_log_probs_batch, 1
            )

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            goals_batch = self._flatten_helper(T, N, goals_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_goal_log_probs_batch = self._flatten_helper(
                T, N, old_goal_log_probs_batch
            )

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                goals_batch,
                masks_batch,
                old_goal_log_probs_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


@baseline_registry.register_trainer(name="odometry")
class OdomTrainer(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self._static_encoder = False
        self.agent = None
        self.envs = None
        self.odom = None
        if config is not None:
            logger.info(f"config: {config}")

    def _setup_agent_and_odom(self) -> None:
        logger.add_filehandler(self.config.LOG_FILE)
        self.agent = RandomAgent(self.config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                                 'pointgoal_with_gps_compass', self.envs.num_envs,)

        self.odom = SimpleOdomNet(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            config=self.config.ODOM,
            num_recurrent_layers=self.config.ODOM.num_recurrent_layers,
            lr=self.config.ODOM.lr,
            eps=self.config.ODOM.eps,
        )

        self.odom.to(self.device)

        if self.config.ODOM.pretrained:
            pretrained_state = torch.load(
                self.config.ODOM.pretrained_weights, map_location="cpu"
            )
            self.odom.load_state_dict(
                {
                    k[len("odom."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.odom.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(
        self, rollouts
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        rollouts.to(self.device)
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            actions = self.agent.act(step_observation).unsqueeze(-1)
            (
                _,
                goal_log_probs,
                _,
                recurrent_hidden_states,
            ) = self.odom(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                actions,
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        outputs = self.envs.step([a.item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        goals = torch.tensor([obs['pointgoal_with_gps_compass'] for obs in observations])

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=self.device,
        )


        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            goals,
            goal_log_probs,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_odom(self, config, rollouts):
        t_update_model = time.time()

        goal_loss, prob_loss, dist_entropy = self.odom.update(rollouts, config)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            goal_loss,
            prob_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        odom_config = self.config.ODOM
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_agent_and_odom()
        if self.config.FP16:
            self.half()
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.odom.parameters())
            )
        )

        rollouts = OdomRolloutStorage(
            odom_config.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            odom_config.hidden_size,
            odom_config.num_recurrent_layers * 2,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.odom.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                if odom_config.use_linear_lr_decay:
                    lr_scheduler.step()

                if odom_config.use_linear_clip_decay:
                    self.odom.clip_param = odom_config.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in range(odom_config.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(rollouts)
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                (
                    delta_pth_time,
                    goal_loss,
                    prob_loss,
                    dist_entropy,
                ) = self._update_odom(self.config, rollouts)
                pth_time += delta_pth_time

                # Check to see if there are any metrics
                # that haven't been logged yet
                losses = [goal_loss, prob_loss]
                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["goal", "probability"])},
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

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

            self.envs.close()

    def half(self):
        self.odom, self.odom.optimizer \
            = amp.initialize(self.odom, self.odom.optimizer,
                             opt_level="O1")

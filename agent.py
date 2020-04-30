import argparse
import random

import numpy as np
import torch
import PIL
from gym.spaces import Discrete, Dict, Box

import habitat
import os
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo import Policy, PointNavBaselinePolicy
from habitat_baselines.rl.ddppo.policy.resnet_policy import  PointNavResNetPolicy
from habitat_baselines.common.utils import batch_obs
from habitat import Config
from habitat.core.agent import Agent
from efficientnet_policy import PointNavEfficientNetPolicy

from slam_tracker import SLAM_TRACKER

def get_defaut_config():
    c = Config()
    c.INPUT_TYPE = "depth"
    c.MODEL_PATH = "data/new_checkpoints/noisy.pth"
    c.RL.PPO.hidden_size = 512
    c.RANDOM_SEED = 7
    c.TORCH_GPU_ID = 0
    return c


class EfficientNetAgent(Agent):
    def __init__(self, config: Config):
        spaces = {
            "pointgoal": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }

        if config.INPUT_TYPE in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 1),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH, 3),
                dtype=np.uint8,
            )
        observation_spaces = Dict(spaces)

        action_spaces = Discrete(4)

        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        self.hidden_size = config.RL.PPO.hidden_size

        random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

        self.actor_critic = PointNavEfficientNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            goal_sensor_uuid=config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs=False,
            # "rgb" in self.envs.observation_spaces[0].spaces,
            pretrained=False,
            finetune=False,
        )
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k.replace("actor_critic.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None
        self._slam_tracker = SLAM_TRACKER()
        self._slam_tracker.init(self.device)
        self._slam_reset = False
 
    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1, self.hidden_size, device=self.device
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )
        self._slam_reset = True

    def act(self, observations):
        self._slam_tracker.update(observation, action)
         self._slam_reset == True:
            self._slam_reset = False

        else:
            observation['pointgoal'] =  self._slam_tracker.get_estimate_goal(observation)

        batch = batch_obs([observations])
        for sensor in batch:
            batch[sensor] = batch[sensor].to(self.device)

        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks = torch.ones(1, 1, device=self.device)
            self.prev_actions.copy_(actions)
        return actions[0][0].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    parser.add_argument("--model-path", default="", type=str)
    args = parser.parse_args()

    config = get_config('ddppo_tamer_pointnav.yaml',
                ['BASE_TASK_CONFIG_PATH', config_paths]).clone()
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path

    config.RANDOM_SEED = 7
    config.freeze()

    agent = EfficientNetAgent(config)
    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()

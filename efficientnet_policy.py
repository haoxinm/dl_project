import numpy as np
import torch
import torch.nn as nn
from habitat_baselines.common.utils import Flatten
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo import Net, Policy
from efficientnet_pytorch.model import efficientnet
import torchvision.transforms as transforms
from torch.nn import functional as F


class PointNavEfficientNetPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid="pointgoal",
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        backbone="efficientnet-b7",
        normalize_visual_inputs=False,
        pretrained=None,
        finetune=True,
    ):
        super().__init__(
            PointNavEfficientNetNet(
                observation_space=observation_space,
                action_space=action_space,
                goal_sensor_uuid=goal_sensor_uuid,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                normalize_visual_inputs=normalize_visual_inputs,
                pretrained=pretrained,
                finetune=finetune,
            ),
            action_space.n,
        )


class EfficientNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        hidden_size=512,
        backbone_name='efficientnet-b7',
        pretrained=False,
        finetune=False,
        normalize_visual_inputs=False,
    ):
        super().__init__()

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = efficientnet(backbone_name, pretrained, input_channels, finetune, hidden_size)

            self.output_shape = hidden_size

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        return x


class PointNavEfficientNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        normalize_visual_inputs,
        pretrained=False,
        finetune=False,
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        self._n_input_goal = (
            observation_space.spaces[self.goal_sensor_uuid].shape[0] + 1
        )
        self.tgt_embeding = nn.Linear(self._n_input_goal, 32)
        self._n_input_goal = 32

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        self.visual_encoder = EfficientNetEncoder(
            observation_space,
            hidden_size=hidden_size,
            backbone_name=backbone,
            pretrained=pretrained,
            finetune=finetune,
            normalize_visual_inputs=normalize_visual_inputs,
        )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_tgt_encoding(self, observations):
        goal_observations = observations[self.goal_sensor_uuid]
        goal_observations = torch.stack(
            [
                goal_observations[:, 0],
                torch.cos(-goal_observations[:, 1]),
                torch.sin(-goal_observations[:, 1]),
            ],
            -1,
        )

        return self.tgt_embeding(goal_observations)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            x.append(visual_feats)

        tgt_encoding = self.get_tgt_encoding(observations)
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        x += [tgt_encoding, prev_actions]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
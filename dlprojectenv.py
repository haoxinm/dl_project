#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type

import habitat
import torch
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.environments import NavRLEnv


# def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
#     r"""Return environment class based on name.
#
#     Args:
#         env_name: name of the environment.
#
#     Returns:
#         Type[habitat.RLEnv]: env class.
#     """
#     return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="DLProjectRLENV")
class DLProjectRLENV(NavRLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        self._follower = self.get_shortest_path_follower()

    def get_shortest_path_follower(self):
        goal_radius = self.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = get_config().SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestPathFollower(self.habitat_env.sim, goal_radius, False)
        follower.mode = "exact_gradient"
        return follower

    def get_best_action(self):
        best_action = self._follower.get_next_action(
                self.habitat_env.current_episode.goals[0].position
        )
        if best_action is None:
            best_action = HabitatSimActions.STOP
        return [torch.tensor(best_action)]
            

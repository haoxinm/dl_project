import os
import random
import sys
import time
from math import pi, sqrt, sin, cos, asin, acos

import numpy as np
import orbslam2
import PIL
import requests
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from habitat_baselines.slambased.mappers import DirectDepthMapper

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.config.default import get_config as cfg_baseline
from habitat_baselines.slambased.reprojection import angle_to_pi_2_minus_pi_2 
from habitat_baselines.slambased.reprojection import (
    get_direction,
    get_distance,
    habitat_goalpos_to_mapgoal_pos,
    homogenize_p,
    planned_path2tps,
    project_tps_into_worldmap,
)
from habitat_baselines.slambased.utils import generate_2dgrid

class SLAM_TRACKER():
    def distance_to_initial(self, current_position):
        initial_position = self.initial_position[0]
        return self.map_cell_size * (current_position - initial_position)

    def distance_angle_to_axis(self, position):
        distance = position[0]
        angle = position[1]
        cos_ang = cos(angle)
        sin_ang = sin(angle)
        x = distance * cos_ang
        y = distance * sin_ang
        return torch.tensor([x, y], device=self.device)


    def orientation_to_angle(self, orientation):
        cos_val = orientation[0][0]
        sin_val = orientation[1][0]
        angle = torch.atan2(sin_val, cos_val)
        # if sin_val < - 0.00001:
        #     angle = - angle

        return angle

    def distance(self, x, y):
        return (x - y).pow(2).sum().sqrt()


    def estimate_distance_to_goal(self, observations):

        distance_to_initial = self.distance_to_initial(self.get_position_on_map()[0])
        initial_target_distance = self.distance_angle_to_axis(self.initial_goal_position)
        # new_target_distance = - initial_target_distance + distance_to_initial
        new_target_distance_val = self.distance(distance_to_initial, initial_target_distance)
        AB = self.map_cell_size * self.distance(self.get_position_on_map()[0], self.initial_position[0])
        AC = self.distance(initial_target_distance, 0.)
        BC = new_target_distance_val
        self_angle = self.orientation_to_angle(self.get_orientation_on_map())

        cos_theta = 1.
        if BC > 0.:
            cos_theta = AC * AC + BC * BC - AB * AB
            cos_theta = cos_theta / (2. * AC * BC)
        theta = acos(cos_theta)
        if self.initial_goal_position[1] > 0:
            gamma = theta + self.initial_goal_position[1]
        else:
            gamma = - theta + self.initial_goal_position[1]
        estimate_angle = gamma - self_angle
        if estimate_angle < - pi:
            estimate_angle = estimate_angle + pi
        elif estimate_angle > pi:
            estimate_angle = estimate_angle - pi
        return [new_target_distance_val, estimate_angle - 0]

    def update_accumate_position(self, action):
        new_position = self.get_position_on_map()[0]
        if action == HabitatSimActions.MOVE_FORWARD: # Move Forward
            difference = new_position - self.last_position
            self.accumlate_position = self.accumlate_position + difference
        else:
            pass
        self.last_position = new_position


    def init_pose6d(self):
        return torch.eye(4).float().to(self.device)

    def init_map2d(self):
        return (
            torch.zeros(
                1, 1, self.map_size_in_cells(), self.map_size_in_cells()
            )
            .float()
            .to(self.device)
        )

    def map_size_in_cells(self):
        return int(self.map_size_meters / self.map_cell_size)

    def init(self, device):
        config = cfg_baseline().ORBSLAM2
        config._immutable(False)
        config.CAMERA_HEIGHT = 0.88


        config.H_OBSTACLE_MIN = (
            0.3 * config.CAMERA_HEIGHT
        )
        config.H_OBSTACLE_MAX = (
            1.0 * config.CAMERA_HEIGHT
        )
        self.slam_vocab_path = config.SLAM_VOCAB_PATH
        # print(self.slam_vocab_path)
        assert os.path.isfile(self.slam_vocab_path)
        self.slam_settings_path = config.SLAM_SETTINGS_PATH
        # print(self.slam_settings_path)
        assert os.path.isfile(self.slam_settings_path)
        self.slam = orbslam2.System(
             self.slam_vocab_path, self.slam_settings_path, orbslam2.Sensor.RGBD
         )
        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.device = device
        # print(self.device)
        self.map_size_meters = config.MAP_SIZE
        self.map_cell_size = config.MAP_CELL_SIZE
        self.pos_th = config.DIST_REACHED_TH
        self.next_wp_th = config.NEXT_WAYPOINT_TH
        self.angle_th = config.ANGLE_TH
        self.obstacle_th = config.MIN_PTS_IN_OBSTACLE
        self.depth_denorm = config.DEPTH_DENORM
        self.planned_waypoints = []
        self.mapper = DirectDepthMapper(
            camera_height=config.CAMERA_HEIGHT,
            near_th=config.D_OBSTACLE_MIN,
            far_th=config.D_OBSTACLE_MAX,
            h_min=config.H_OBSTACLE_MIN,
            h_max=config.H_OBSTACLE_MAX,
            map_size=config.MAP_SIZE,
            map_cell_size=config.MAP_CELL_SIZE,
            device=device,
        )
        self.slam_to_world = 1.0
        self.timestep = 0.1
        self.timing = False
        self.reset()
        return

    def reset(self):
        self.steps = 0

        self.tracking_is_OK = False
        self.waypointPose6D = None
        self.unseen_obstacle = False
        self.action_history = []
        self.planned_waypoints = []
        self.map2DObstacles = self.init_map2d()
        n, ch, height, width = self.map2DObstacles.size()
        self.coordinatesGrid = generate_2dgrid(height, width, False).to(
            self.device
        )
        self.pose6D = self.init_pose6d()
        self.action_history = []
        self.pose6D_history = []
        self.position_history = []
        self.planned2Dpath = torch.zeros((0))
        #self.slam.shutdown()
        #del self.slam
        #self.slam.set_use_viewer(False)
        self.slam.reset()
        self.cur_time = 0
        self.toDoList = []
        self.waypoint_id = 0
        self.initial_position = None
        self.initial_angle = None
        self.initial_6D = None
        self.initial_goal_position = None
        self.estimate_target_pos = None
        self.set_initial = False
        self.last_position = None
        self.accumlate_position = None

        if self.device != torch.device("cpu"):
            torch.cuda.empty_cache()
        return

    def get_position_on_map(self, do_floor=True):
        return project_tps_into_worldmap(
            self.pose6D.view(1, 4, 4),
            self.map_cell_size,
            self.map_size_meters,
            do_floor,
        )

    def get_orientation_on_map(self):
        self.pose6D = self.pose6D.view(1, 4, 4)
        return torch.tensor(
            [
                [self.pose6D[0, 0, 0], self.pose6D[0, 0, 2]],
                [self.pose6D[0, 2, 0], self.pose6D[0, 2, 2]],
            ]
        )

    def rgb_d_from_observation(self, habitat_observation):
        rgb = habitat_observation["rgb"]
        depth = None
        if "depth" in habitat_observation:
            depth = self.depth_denorm * habitat_observation["depth"]
        return rgb, depth


    def update_internal_state(self, observation):
        self.steps += 1
        self.cur_time += self.timestep
        rgb, depth = self.rgb_d_from_observation(observation)
        t = time.time()
        try:
            self.slam.process_image_rgbd(rgb, depth, self.cur_time)
            if self.timing:
                print(time.time() - t, "ORB_SLAM2")
            self.tracking_is_OK = str(self.slam.get_tracking_state()) == "OK"
        except BaseException:
            print("Warning!!!! ORBSLAM processing frame error")
            self.tracking_is_OK = False
        if not self.tracking_is_OK:
            self.reset()
        t = time.time()
        if self.tracking_is_OK:
            trajectory_history = np.array(self.slam.get_trajectory_points())
            self.pose6D = homogenize_p(
                torch.tensor(trajectory_history[-1], device=self.device)[1:]
                .view(3, 4)
            ).view(1, 4, 4)
            self.trajectory_history = trajectory_history
            if len(self.position_history) > 1:
                previous_step = get_distance(
                    self.pose6D.view(4, 4),
                    torch.tensor(self.position_history[-1], device=self.device)
                    .view(4, 4),
                )
                if len(self.action_history) > 0 and self.action_history[-1] == HabitatSimActions.MOVE_FORWARD:
                    self.unseen_obstacle = (
                        previous_step.item() <= 0.001
                    )  # hardcoded threshold for not moving
        current_obstacles = self.mapper(
            torch.tensor(depth, device=self.device).squeeze(), self.pose6D
        ).to(self.device)

        self.current_obstacles = current_obstacles
        self.map2DObstacles = torch.max(
            self.map2DObstacles, current_obstacles.unsqueeze(0).unsqueeze(0)
        )
        return True

    def update(self, observation, action):
        if self.timing:
            print(time.time() - t, "Mapping")
        if self.set_initial == False and self.initial_position == None:
            self.set_initial = True
            self.initial_position = self.get_position_on_map()
            self.initial_goal_position = observation['pointgoal']

        if self.initial_angle == None:
            self.initial_angle = 0

        if self.initial_6D == None:
            self.initial_6D = self.pose6D

        if self.last_position == None:
            self.last_position = torch.tensor([0, 0]).to(self.device)
        if self.accumlate_position == None:
            self.accumlate_position = torch.tensor([0, 0]).to(self.device)
        cc = 0
        update_is_ok = self.update_internal_state(observation)
        while not update_is_ok:
            update_is_ok = self.update_internal_state(observation)
            cc += 1
            if cc > 2:
                break
        self.position_history.append(
            self.pose6D.detach().cpu().numpy().reshape(1, 4, 4)
        )

        if self.last_position == None:
            self.last_position = torch.tensor([0, 0]).to(self.device)
        if self.accumlate_position == None:
            self.accumlate_position = torch.tensor([0, 0]).to(self.device)
        if self.set_initial == False and self.initial_position == None:
            self.initial_position = self.get_position_on_map()
            self.set_initial = True
            self.initial_goal_position = observation['pointgoal']

        self.update_accumate_position(action)
        return

    def get_estimate_goal(self, observation):
        return self.estimate_distance_to_goal(observation)

import numpy as np
from gym.envs.registration import register
from highway_env import utils
from highway_env.envs import HighwayEnvFast, HighwayEnv, IntersectionEnv, RoundaboutEnv, UTurnEnv, TwoWayEnv, MergeEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from classic_control_env import *
#from classic_control_env.mountain_car_env import MountainCarEnv
#from gym.envs.classic_control import MountainCarEnv
from classic_control_env.cartpole import CartPoleEnv

class CartPoleWithCost(CartPoleEnv):
    #        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    # overriden method for GAN
    def __init__(self):
        super().__init__()
        default_parameter_ranges = {
            'gravity': (9.0, 10.0),
            'masscart':  (0.5, 2),
            'masspole':  (0.05, 0.5),
            'length': (0.4, 0.6),
        } #ranges to use when there is no GAN output
        default_parameter_values = {}
        for param, param_range in default_parameter_ranges.items():
            default_parameter_values[param] = self.np_random.uniform(low= param_range[0], high=param_range[1])
        self.update_params(default_parameter_values)


    def _cost(self, state):
        pole_angle = state[2]
        cart_position = state[0]
        # TODO -
        if abs(pole_angle) >= self.theta_threshold_radians or abs(cart_position) >= self.x_threshold:
            return 1
        else:
            return 0


    # overriden method for gan
    def update_params(self, params_dct):
        for param, param_value in params_dct.items():
            setattr(self, param, param_value)
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = (self.masspole + self.length)

    # overriden method for gan
    def reset(self, gan_output=None):
        if gan_output:
            self.update_params(gan_output)
        # TODO - 23.01 - Added this self.init() for getting new updated random params
        else:
            self.__init__()
        # else - it will keep the original default params values.
        self.state = self.np_random.uniform(low= -0.05, high= 0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)


class HighwayEnvFastNoNormalization(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    # Overriding method
    # TODO -
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "OccupancyGrid",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                "grid_step": [5, 5],
                "absolute": False
            },
            "duration": 50,
            # TODO - DEFINE AS CONFIG PARAM - vehicles_count - V
            "vehicles_count": 20,
            # TODO - DEFINE AS CONFIG PARAM - vehicles_density - V
            "vehicles_density": 2,
            # TODO - DEFINE AS CONFIG PARAM - ego_spacing - V
            # TODO - DEFINE AS CONFIG PARAM - np_random

        })
        return cfg

        # Overriding method
    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        # TODO - DEFINE AS CONFIG PARAM - speed_limit - V
        speed_limit = self.config.get('speed_limit', 30)
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"],
                                                                   speed_limit= speed_limit),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

class HighwayEnvSimple(HighwayEnv):
    """
    A simplified variant of highway-v0 with a less challenging environment:
        - lower number of vehicles
        - lower density of vehicles
        - simpler observation features
    """
    # Overriding method
    # TODO -
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "OccupancyGrid",
                "vehicles_count": 5,  # Reduced the number of vehicles
                "features": ["presence", "x", "y", "vx", "vy"], # No cosx, etc.
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                "grid_step": [5, 5],
                "absolute": False
            },
            "duration": 50,
            "vehicles_count": 10,  # Reduce the total number of vehicles
            "vehicles_density": 1,  # Reduce the density of vehicles (per timestep ?)
        })
        return cfg


        # Overriding method
    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        # TODO - DEFINE AS CONFIG PARAM - speed_limit - V
        speed_limit = self.config.get('speed_limit', 30)
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"],
                                                                   speed_limit= speed_limit),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class IntersectionEnvNoNormalization(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "flatten": False,
                "observe_intentions": False,
                "see_behind": True
            }
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

class RoundaboutEnvNoNormalization(RoundaboutEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
                "see_behind": True
            }
        })
        return config

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

class UTurnEnvNoNormalization(UTurnEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "see_behind": True
            },
        })
        return config

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

class TwoWayEnvNoNormalization(TwoWayEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "see_behind": True
            },
        })
        return config

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        if len(neighbours) == 1:
            return 0
        reward = self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1) \
            + self.config["left_lane_reward"] \
                * (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (len(neighbours) - 1)
        return reward

class MergeEnvNoNormalization(MergeEnv):

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "see_behind": True
            },
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)






class EgoMergeEnvNoNormalization(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "see_behind": True
            },
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

    def _make_vehicles(self) -> None:

        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(110, 0),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        merging_v = other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=20)
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle


    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.config["lane_change_reward"],
                         1: 0,
                         2: self.config["lane_change_reward"],
                         3: 0,
                         4: 0}
        reward = self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * self.vehicle.lane_index[2] / 1 \
            + self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                speed_reward = 0 if vehicle.target_speed == 0 else (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                reward += self.config["merging_speed_reward"] * speed_reward

        return utils.lmap(action_reward[action] + reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

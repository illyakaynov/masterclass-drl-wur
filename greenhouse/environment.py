
import gym
from gym import spaces

from itertools import product

from greenhouse.simulation import Greenhouse, GreenhouseAction
from greenhouse.weather import Weather, WeatherDefault
import numpy as np


class GreenhouseEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or dict()
        self.max_episode_time = self.config.get('max_episode_time', 7 * 24 * 60)

        greenhouse_config = self.config.get('greenhouse_config', dict())
        self.greenhouse_model = Greenhouse(weather_model=Weather(), **greenhouse_config)

        sample_obs = self.greenhouse_model.reset()
        high = np.array([np.inf] * sample_obs.size())
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Define the range of values avalilable for actions
        ranges = [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
        # Create a cartesian product of all of the actions
        possible_actions = list(product(*ranges))
        # Create a mapping from a number to a vector representation of an action and back
        self.action_vec_to_num = {}
        self.action_num_to_vec = {}
        for i, action in enumerate(possible_actions):
            self.action_vec_to_num[action] = i
            self.action_num_to_vec[i] = action

        self.action_space = spaces.Discrete(len(self.action_vec_to_num))

    def reset(self):
        obs_np = self.greenhouse_model.reset().to_numpy()
        return obs_np

    def step(self, action):
        # if action is an integer covert it into a vector form
        if isinstance(action, int):
            action = self.action_num_to_vec[action]

        # create a greenhouse action
        greenhouse_action = GreenhouseAction(
            heater=action[0],
            window=action[1],
            vapor_supply=action[2],
            CO2_supply=action[3],
            light=action[4],
        )

        # perform a step in a simulation
        greenhouse_state, reward, reward_dict = self.greenhouse_model.step(greenhouse_action)

        done = self.greenhouse_model.time >= self.max_episode_time
        info = {**reward_dict}

        return greenhouse_state.to_numpy(), reward, done, info
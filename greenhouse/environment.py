
import gym
from gym import spaces

from itertools import product

from greenhouse.simulation import Greenhouse, GreenhouseAction
from greenhouse.weather import Weather, WeatherDefault
import numpy as np


class GreenhouseEnv(gym.Env):
    def __init__(self, config=None):
        super(GreenhouseEnv, self).__init__()
        self.config = config or dict()

        self.time = 0

        self.greenhouse_model = Greenhouse(weather_model=Weather())
        sample_obs = self.greenhouse_model.reset()

        high = np.array([np.inf] * sample_obs.size())
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        ranges = [
            [0, 0.5],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
        possible_actions = list(product(*ranges))
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
        if isinstance(action, int):
            action = self.action_num_to_vec[action]

        greenhouse_action = GreenhouseAction(
            heater=action[0],
            window=action[1],
            vapor_supply=action[2],
            CO2_supply=action[3],
        )

        greenhouse_state, reward = self.greenhouse_model.step(greenhouse_action)

        sim_max_minutes = 7 * 24 * 60

        done = self.greenhouse_model.time >= sim_max_minutes
        info = {}

        return greenhouse_state.to_numpy(), reward, done, info
import gym
from gym import spaces
import numpy as np

from QLearning.GridWorld.state import State

class GridEnv(gym.Env):
    ACTIONS = {0: "left", 1: "up", 2: "right", 3: "down"}
    REWARDS = {'trap': -1, 'goal': +1, 'timestep': -0.1}

    def __init__(self, layout_id=0, terminate_after=None):
        super().__init__()
        self.state = State(layout_id=layout_id)
        self.time = 0
        self.end_time = 20
        max_state = 10 * self.state.shape[0] + self.state.shape[1]
        self.observation_space = spaces.Discrete(max_state)
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.terminate_after = terminate_after

    def reset(self):
        self.state.reset()
        self.time = 0
        return self.get_obs()

    def apply_action(self, action):
        if action == 0:
            self.state.move(dy=-1)
        elif action == 1:
            self.state.move(dx=-1)
        elif action == 2:
            self.state.move(dy=+1)
        elif action == 3:
            self.state.move(dx=+1)
        else:
            raise ValueError("Unknown action {}".format(action))

    def get_obs(self):
        x, y = self.state.get_player_pos()
        return self.coord_to_state(x, y)

    def step(self, action):
        done = False

        self.apply_action(action)
        reward = self.compute_reward()
        obs = self.get_obs()
        self.time += 1

        x, y = self.state.get_player_pos()
        if self.state.get_state(x, y) == self.state.CELLS['trap']:
            done = True
        if self.state.get_state(x, y) == self.state.CELLS['goal']:
            done = True

        if self.terminate_after and self.time >= self.terminate_after:
            done = True

        return obs, reward, done, {}

    def compute_reward(self):
        x, y = self.state.get_player_pos()
        cell_value = self.state.get_state(x, y)
        reward = 0

        if cell_value == self.state.CELLS['trap']:
            reward += self.REWARDS['trap']
        elif cell_value == self.state.CELLS['goal']:
            reward += self.REWARDS['goal']
        else:
            reward += self.REWARDS['timestep']

        return reward

    def get_state(self):
        return self.state

    def set_state(self, state):
        x, y = self.state_to_coord(state)
        self.state.set_state(x, y)

    def get_allowed_player_states(self):
        x, y = self.state.get_allowed_player_positions()
        return self.coord_to_state(x, y)

    def get_allowed_states(self):
        x, y = self.state.get_allowed_positions()
        return self.coord_to_state(x, y)

    def coords_to_state(self, x: np.ndarray, y: np.ndarray) -> list:
        coord_or_coords = (x * 10 + y)
        return coord_or_coords.tolist()

    def coord_to_state(self, x: int, y: int) -> int:
        coord_or_coords = (x * 10 + y)
        return coord_or_coords


    def state_to_coord(self, state):
        y = state % 10
        x = (state - y) // 10
        return x, y

    def get_terminal_states(self):
        x, y = self.state.get_terminal_positions()
        return self.coord_to_state(x, y)

    def get_allowed_actions(self):
        return list(self.ACTIONS.keys())

    def get_action_description(self):
        return self.ACTIONS

    def render(self, mode="txt"):
        self.state.render()

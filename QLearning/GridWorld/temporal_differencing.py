import numpy as np
from QLearning.GridWorld.monte_carlo import ValueAgent


class TDValue(ValueAgent):
    def __init__(self, env):
        self.trajectory = []
        self.value_fn = {}
        self.returns = {}
        self.policy = {}
        self.alpha = 0.01
        self.threshold = 1e-4
        self.episode_counter = 0
        self.step_counter = 0
        self.gamma = 0.9
        self.env = env
        self.n_actions = env.action_space.n
        self.reset_values()
        self.episode_count = 0
        self.epsilon = 0.2

    def get_type(self):
        return "value"

    def reset(self):
        self.episode_count += 1

    def reset_values(self):
        for state in self.env.get_allowed_states():
            self.returns[state] = []
            self.value_fn[state] = 0
            self.policy[state] = np.random.choice(self.env.get_allowed_actions())
        for state in self.env.get_terminal_states():
            self.value_fn[state] = 0

        self.episode_count = 0

    def compute_action(self, obs):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.env.get_allowed_actions())
        else:
            action = self.compute_greedy_action(obs)
        return action

    def compute_greedy_action(self, obs):
        next_state_values = self.get_values_or_q_values(obs)
        return np.argmax(next_state_values)

    def update(self, obs, action, next_obs, reward, done):
        self.value_fn[obs] = self.value_fn[obs] + self.alpha * (reward + self.gamma * self.value_fn[next_obs] - self.value_fn[obs])

    def policy_improvement(self):
        for state in self.env.get_allowed_player_states():
            next_state_values = []
            for action in self.env.get_allowed_actions():
                self.env.reset()
                self.env.set_state(state)
                next_state, reward, _, _ = self.env.step(action)
                next_state_values.append(reward + self.gamma * self.value_fn[next_state])
            self.policy[state] = np.argmax(next_state_values)

    def learn_until_convergence(self):
        n_iter = 0
        delta = 1e10
        while delta > self.threshold:
            done = False
            obs = self.env.reset()
            #     env.set_state(np.random.choice(env.get_allowed_player_states()))
            while not done:
                action = self.compute_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.update(obs, action, next_obs, reward, done)
                obs = next_obs

            self.reset()
            n_iter += 1
            if n_iter >= 2000:
                break

        return delta

    def learn_one_iteration(self):
        print(self.learn_until_convergence())

class TDValueRandom(ValueAgent):
    def __init__(self, env):
        self.trajectory = []
        self.value_fn = {}
        self.returns = {}
        self.alpha = 0.01
        self.threshold = 1e-4
        self.episode_counter = 0
        self.step_counter = 0
        self.gamma = 0.9
        self.env = env
        self.n_actions = env.action_space.n
        self.reset_values()
        self.episode_count = 0

    def get_type(self):
        return "value"

    def reset(self):
        self.episode_count += 1

    def reset_values(self):
        for state in self.env.get_allowed_states():
            self.returns[state] = []
            self.value_fn[state] = 0
        for state in self.env.get_terminal_states():
            self.value_fn[state] = 0
        self.episode_count = 0

    def compute_action(self, obs):
        action = np.random.choice(self.env.get_allowed_actions())
        return action

    def compute_greedy_action(self, obs):
        next_state_values = self.get_values_or_q_values(obs)
        return np.argmax(next_state_values)

    def update(self, obs, action, next_obs, reward, done):
        self.value_fn[obs] = self.value_fn[obs] + self.alpha * (reward + self.gamma * self.value_fn[next_obs] - self.value_fn[obs])

    def learn_until_convergence(self):
        n_iter = 0
        delta = 1e10
        while delta > self.threshold:
            done = False
            obs = self.env.reset()
            #     env.set_state(np.random.choice(env.get_allowed_player_states()))
            while not done:
                action = self.compute_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.update(obs, action, next_obs, reward, done)
                obs = next_obs

            self.reset()
            n_iter += 1
            if n_iter >= 2000:
                break

        return delta

    def learn_one_iteration(self):
        print(self.learn_until_convergence())
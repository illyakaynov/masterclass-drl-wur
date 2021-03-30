import numpy as np

from QLearning.GridWorld.grid_env import GridEnv
from collections import defaultdict

class Agent:
    def compute_action(self, obs):
        pass

    def update(self, *args, **kwargs):
        pass

    def policy_evaluation(self):
        pass

    def policy_improvement(self):
        pass

    def learn_one_iteration(self):
        pass

class MonteCarloValue(Agent):

    def __init__(self, env, epsilon=0.2):
        self.trajectory = []
        self.value_fn = {}
        self.returns = {}
        self.policy = {}
        self.gamma = 0.9
        self.env = env
        self.epsilon = epsilon
        self.reset_values()

    def get_type(self):
        return 'value'

    def reset(self):
        self.policy_evaluation()
        self.policy_improvement()
        self.trajectory = []

    def reset_values(self):
        allowed_actions = self.env.get_allowed_actions()
        for state in self.env.get_allowed_states():
            self.returns[state] = []
            self.value_fn[state] = np.random.uniform()

        for state in self.env.get_allowed_player_states():
            self.policy[state] = np.random.choice(allowed_actions)

    def compute_action(self, obs):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.env.get_allowed_actions())
        else:
            action = self.policy[obs]
        return action

    def compute_greedy_action(self, obs):
        return self.policy[obs]

    def policy_evaluation(self):
        return_ = 0
        visited_obs = set()
        for obs, action, next_obs, reward, done in self.trajectory[::-1]:
            return_ = return_ + reward
            if obs not in visited_obs:
                self.returns[obs].append(return_)
                self.value_fn[obs] = np.mean(self.returns[obs])
                visited_obs.add(obs)

    def policy_improvement(self):
        for state in self.env.get_allowed_player_states():
            next_state_values = []
            for action in self.env.get_allowed_actions():
                self.env.reset()
                self.env.set_state(state)
                next_state, reward, _, _ = self.env.step(action)
                next_state_values.append(reward + self.gamma * self.value_fn[next_state])
            self.policy[state] = np.argmax(next_state_values)

    def update(self, obs, action, next_obs, reward, done):
        self.trajectory.append((obs, action, next_obs, reward, done))

class MonteCarloQValue(Agent):
    def __init__(
            self, n_states, n_actions, epsilon=0.2, alpha=0.1, gamma=0.9
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.n_actions = n_actions
        self.n_states = n_states

        self.sum_rewards = 0
        self.return_ = 0

        self.trajectory = []
        self.returns = defaultdict(list)

        self.q_table = np.zeros((self.n_states, self.n_actions), np.float)

        self.policy = {}

    def get_type(self):
        return 'q_value'

    def reset(self):
        self.sum_rewards = 0
        self.return_ = 0
        self.policy_evaluation()
        self.trajectory = []

    def reset_values(self):
        self.q_table = np.zeros((self.n_states, self.n_actions), np.float)

    def compute_greedy_action(self, obs):
        return self.q_table[obs, :].argmax()

    def compute_action(self, obs):
        action_probs = self.policy.get(obs, None)
        action = np.random.choice(self.n_actions, p=action_probs)
        return action

    def update(self, obs, action, next_obs, reward, done):
        self.trajectory.append((obs, action, next_obs, reward, done))

    def policy_evaluation(self):
        return_ = 0
        visited_obs_action_pairs = set()
        for obs, action, next_obs, reward, done in self.trajectory[::-1]:
            return_ = return_ + reward
            if (obs, action) not in visited_obs_action_pairs:
                self.returns[(obs, action)].append(return_)
                self.q_table[obs, action] = np.mean(self.returns[(obs, action)])
                visited_obs_action_pairs.add((obs, action))
                greedy_action = self.q_table[obs, :].argmax()
                action_probs = []
                for a in range(self.n_actions):
                    if a == greedy_action:
                        action_prob = 1 - self.epsilon + self.epsilon / self.n_actions
                    else:
                        action_prob = self.epsilon / self.n_actions
                    action_probs.append(action_prob)
                self.policy[obs] = action_probs


class MonteCarloValueRandom(Agent):

    def __init__(self, env):
        self.trajectory = []
        self.value_fn = {}
        self.returns = {}
        self.alpha = 0.001
        self.threshold = 1e-4
        self.episode_counter = 0
        self.gamma = 0.9
        self.env = env
        self.n_actions = env.action_space.n
        self.reset_values()
        self.episode_count = 0

    def get_type(self):
        return 'value'

    def reset(self):
        self.policy_evaluation()
        self.trajectory = []
        self.episode_count += 1

    def reset_values(self):
        for state in self.env.get_allowed_states():
            self.returns[state] = []
            self.value_fn[state] = 0
        for state in self.env.get_terminal_states():
            self.value_fn[state] = 0

    def compute_action(self, obs):
        action = np.random.choice(self.env.get_allowed_actions())
        return action

    def compute_greedy_action(self, obs):
        next_state_values = []
        for action in range(self.n_actions):
            self.env.reset()
            self.env.set_state(obs)
            next_state, reward, _, _ = self.env.step(action)
            next_state_values.append(reward + self.gamma * self.value_fn[next_state])
        return np.argmax(next_state_values)

    def policy_evaluation(self):
        self.generate_trajectory()
        return_ = 0
        visited_obs = set()
        delta = 0
        for obs, action, next_obs, reward, done in self.trajectory[::-1]:
            return_ = self.gamma * return_ + reward
            if obs not in visited_obs:
                old_value = self.value_fn[obs]
                self.returns[obs].append(return_)
                self.value_fn[obs] = self.value_fn[obs] + self.alpha * (return_ - self.value_fn[obs])
                delta = max(delta, np.abs(old_value - self.value_fn[obs]))
                visited_obs.add(obs)
        return delta

    def policy_improvement(self):
        ...

    def update(self, obs, action, next_obs, reward, done):
        self.trajectory.append((obs, action, next_obs, reward, done))

    def generate_trajectory(self):
        obs = self.env.reset()
        done = False
        while not done:
            action = self.compute_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            self.update(obs, action, next_obs, reward, done)
            obs = next_obs

    def learn_until_convergence(self):
        delta = 1e10
        while delta > self.threshold:
            self.generate_trajectory()
            delta = self.policy_evaluation()
            self.trajectory = []
            print(delta)
        return delta

    def learn_one_iteration(self):
        print(self.learn_until_convergence())



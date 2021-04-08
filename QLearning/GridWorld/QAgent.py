import numpy as np

# For plotting metrics
all_epochs = []
all_penalties = []


class QAgent:
    def __init__(
        self, n_states, n_actions, epsilon=0.2, alpha=0.1, gamma=0.9, learn=True
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.n_actions = n_actions
        self.n_states = n_states
        self.learn = learn

        self.sum_rewards = 0
        # self.return_ = 0
        self.episode_count = 0


        self.q_table = np.zeros((self.n_states, self.n_actions), np.float)

    def get_type(self):
        return 'q_value'

    def reset(self):
        self.sum_rewards = 0
        # self.return_ = 0
        self.episode_count += 1

    def reset_values(self):
        self.q_table = np.zeros((self.n_states, self.n_actions), np.float)
        self.episode_count = 0


    def compute_action(self, obs):
        if self.learn and (np.random.random() < self.epsilon):
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.q_table[obs, :])
        return action

    def compute_greedy_action(self, obs):
        return np.argmax(self.q_table[obs, :])

    def get_values_or_q_values(self, obs):
        return self.q_table[obs]

    def update(self, obs, action, next_obs, reward, done):
        if self.learn:
            old_q = self.q_table[obs, action]
            td_update = (
                self.gamma * np.max(self.q_table[next_obs, :])
                - self.q_table[obs, action]
            )
            self.q_table[obs, action] = old_q + self.alpha * (reward + td_update)
            # self.sum_rewards += reward
            # self.return_ += reward * self.gamma

class DoubleQAgent:
    def __init__(
        self, n_states, n_actions, epsilon=0.2, alpha=0.1, gamma=0.9, learn=True
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.n_actions = n_actions
        self.n_states = n_states
        self.learn = learn

        self.sum_rewards = 0

        self.reset()

    def reset(self):
        self.sum_rewards = 0
        self.q_table = np.zeros((self.n_states, self.n_actions), np.float)
        self.q_table_1 = np.zeros((self.n_states, self.n_actions), np.float)

    def compute_action(self, obs):
        if self.learn and (np.random.random() < self.epsilon):
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax((self.q_table[obs, :] + self.q_table_1[obs, :]) / 2)
        return action

    def update(self, obs, action, next_obs, reward, done):
        if self.learn:
            if np.random.randint(2):
                q_table_action = self.q_table
                q_table_value = self.q_table_1
            else:
                q_table_action = self.q_table
                q_table_value = self.q_table_1

            old_q = q_table_action[obs, action]
            td_update = (
                self.gamma * np.max(q_table_value[next_obs, :])
                - q_table_action[obs, action]
            )
            q_table_action[obs, action] = old_q + self.alpha * (reward + td_update)
            self.sum_rewards += reward
import numpy as np


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class RandomPolicy:
    def __init__(self, n_actions, probs=None):
        self.n_actions = n_actions
        self.mapping = {}
        self.probs = probs or [1/self.n_actions for _ in range(self.n_actions)]

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __getitem__(self, key):
        return self.mapping[key]

class DPAgent:
    def __init__(self, env):
        self.policy = {}
        self.value_fn = {}
        self.gamma = 0.9
        self.threshold = 1e-4
        self.env = env
        self.reset_values()

    def reset(self):
        ...

    def compute_action(self, state):
        return self.policy[state]

    def compute_greedy_action(self, state):
        return self.policy[state]

    def update(self, *args, **kwargs):
        ...

    def reset_values(self):
        self.all_actions = self.env.get_allowed_actions()

        for state in self.env.get_allowed_player_states():
            self.policy[state] = np.random.choice(self.all_actions)

        for state in self.env.get_allowed_states():
            self.value_fn[state] = 0

        for terminal_state in self.env.get_terminal_states():
            self.value_fn[terminal_state] = 0

    def get_type(self):
        return 'value'

    def policy_evaluation(self):
        delta = 10
        while delta > self.threshold:
            delta = self.policy_evaluation_one_iteration()
        return delta

                # print(i)
            # policy_stable = self.policy_improvement()


    def policy_evaluation_one_iteration(self):
        delta = 0
        for state in self.env.get_allowed_player_states():
            old_value = self.value_fn[state]
            action = self.policy[state]
            self.env.reset()
            self.env.set_state(state)
            next_state, reward, done, info = self.env.step(action)
            self.value_fn[state] = reward + self.gamma * self.value_fn[next_state]
            delta = max(delta, np.abs(old_value - self.value_fn[state]))
        # for state in self.env.get_allowed_player_states():
        #     self.policy[state] = np.random.choice(self.all_actions)
        print('Policy Evaluation')

        return delta

    def policy_improvement(self):
        policy_stable = True
        for state in self.env.get_allowed_player_states():
            old_action = self.policy[state]
            next_state_values = []
            for action in self.all_actions:
                self.env.reset()
                self.env.set_state(state)
                next_state, reward, _, _ = self.env.step(action)
                next_state_values.append(reward + self.gamma * self.value_fn[next_state])
            self.policy[state] = np.argmax(next_state_values)

            if self.policy[state] != old_action:
                policy_stable = False
        print('Policy Improvement')
        return policy_stable

    def learn_one_iteration(self):
        delta = 1e10
        while delta > self.threshold:
            delta = self.policy_evaluation()
        print(delta)
        policy_stable = self.policy_improvement()
        print(policy_stable)
        return delta, policy_stable

    def get_values_or_q_values(self, state):
        next_state_values = []
        for action in self.all_actions:
            self.env.reset()
            self.env.set_state(state)
            next_state, reward, _, _ = self.env.step(action)
            next_state_values.append(reward + self.gamma * self.value_fn[next_state])
        return next_state_values

class DPRandomAgent:
    def __init__(self, env, probs=None):
        self.value_fn = {}
        self.gamma = 0.9
        self.threshold = 1e-4
        self.env = env
        self.n_actions = env.action_space.n
        self.probs = probs or [1/self.n_actions for _ in range(self.n_actions)]
        self.reset_values()

    def reset(self):
        ...

    def compute_action(self, state):
        return np.random.randint(self.n_actions)

    def compute_greedy_action(self, state):
        next_state_values = self.get_values_or_q_values(state)
        return np.argmax(next_state_values)

    def get_values_or_q_values(self, state):
        next_state_values = []
        for action in self.all_actions:
            self.env.reset()
            self.env.set_state(state)
            next_state, reward, _, _ = self.env.step(action)
            next_state_values.append(reward + self.gamma * self.value_fn[next_state])
        return next_state_values

    def update(self, *args, **kwargs):
        ...

    def reset_values(self):
        self.all_actions = self.env.get_allowed_actions()

        for state in self.env.get_allowed_states():
            self.value_fn[state] = 0

        for terminal_state in self.env.get_terminal_states():
            self.value_fn[terminal_state] = 0

    def get_type(self):
        return 'value'

    def learn_until_convergence(self):
        delta = 1e10
        while delta > self.threshold:
            delta = self.policy_evaluation()
        print('Learning until convergece')
        return delta

    def policy_evaluation(self):
        delta = 0
        for state in self.env.get_allowed_player_states():
            old_value = self.value_fn[state]
            self.env.reset()
            self.env.set_state(state)
            state_value = 0
            for action in self.env.get_allowed_actions():
                self.env.reset()
                self.env.set_state(state)
                next_state, reward, done, info = self.env.step(action)
                state_value += self.probs[action] * (reward + self.gamma * self.value_fn[next_state])
            self.value_fn[state] = state_value
            delta = max(delta, np.abs(old_value - self.value_fn[state]))


        # for state in self.env.get_allowed_player_states():
        #     self.policy[state] = np.random.choice(self.all_actions)
        return delta

    def learn_one_iteration(self):
        print(self.learn_until_convergence())



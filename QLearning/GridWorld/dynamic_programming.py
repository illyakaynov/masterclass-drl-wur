import numpy as np


class DPAgent:
    def __init__(self, env):
        self.policy = {}
        self.value_fn = {}
        self.gamma = 0.99
        self.threshold = 1e-4
        self.env = env
        self.reset_values()

    def reset(self):
        ...

    def compute_action(self, state):
        return self.policy[state]

    def compute_greedy_action(self, state):
        return self.compute_action(state)

    def update(self, *args, **kwargs):
        ...

    def reset_values(self):
        self.all_actions = self.env.get_allowed_actions()

        for state in self.env.get_allowed_player_states():
            self.policy[state] = np.random.choice(self.all_actions)

        for state in self.env.get_allowed_states():
            self.value_fn[state] = np.random.uniform(0, 1)

        for terminal_state in self.env.get_terminal_states():
            self.value_fn[terminal_state] = 0

    def get_type(self):
        return 'value'

    def learn_until_convergence(self):
        i = 0
        policy_stable = False
        while not policy_stable:
            delta = 1e10
            while delta > self.threshold:
                delta = self.policy_evaluation()
                i += 1
                # print(i)
            policy_stable = self.policy_improvement()


    def policy_evaluation(self):
        delta = 0
        for state in self.env.get_allowed_player_states():
            old_value = self.value_fn[state]
            action = self.policy[state]
            self.env.reset()
            self.env.set_state(state)
            next_state, reward, done, info = self.env.step(action)
            print(next_state, reward)
            self.value_fn[state] = reward + self.gamma * self.value_fn[next_state]
            delta = max(delta, np.abs(old_value - self.value_fn[state]))
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
        return policy_stable

    def learn_one_iteration(self):
        delta = 1e10
        while delta > self.threshold:
            delta = self.policy_evaluation()
        policy_stable = self.policy_improvement()
        return delta, policy_stable





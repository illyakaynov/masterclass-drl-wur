from greenhouse.greenhouse import GreenhouseEnv, GreenhouseObservation, plot_history, GreenhouseAction
import numpy as np

env =  GreenhouseEnv()


class HeuristicAgent:
    def compute_action(self, state):

        time = state[0]
        hour = (time % (24 * 12)) / 12

        CO2 = state[3]
        if (hour > 8) & (hour < 20) & (CO2 < 800):
            CO2_supply = 1
        else:
            CO2_supply = 0
        temp = state[1]
        # print(temp)
        if (temp > 28.0):
            window = 1
        else:
            window = 0

        if (temp < 22.0):
            heater = 1
        else:
            heater = 0
        humidity = state[2]
        if (humidity < 80):
            vapor_supply = 1
        else:
            vapor_supply = 0

        action = GreenhouseAction(
            heater=heater,
            window=window,
            vapor_supply=vapor_supply,
            CO2_supply=CO2_supply,
        )
        action = action.to_numpy()
        return action

    def update(self, *args, **kwargs):
        ...

    def finalize_episode(self, *args, **kwargs):
        return {}

if __name__ == '__main__':
    obs = env.reset()
    done = False
    history = []
    agent = HeuristicAgent()
    while not done:
        action = agent.compute_action(obs)
        next_obs, reward, done, info = env.step(action)
        # print(reward)
        history.append(np.concatenate([obs, action]))
        obs = next_obs
    import matplotlib.pyplot as plt
    plot_history(np.asarray(history))
    plt.show()
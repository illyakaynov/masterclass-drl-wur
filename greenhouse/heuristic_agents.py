from greenhouse.simulation import GreenhouseAction
# from QLearning.dqn.agents import Agent

class NoCostHeuristicAgent():
    def compute_action(self, state):
        temp = state[1]
        # print(temp)
        if temp > 28.0:
            window = 1
        else:
            window = 0

        if temp < 25.0:
            heater = 1
        else:
            heater = 0
        humidity = state[2]
        if humidity < 80:
            vapor_supply = 1
        else:
            vapor_supply = 0

        # scatterplot: daylight horizontaly and sun intensity
        # find the correlation between the

        action = GreenhouseAction(
            heater=heater,
            window=window,
            vapor_supply=vapor_supply,
            CO2_supply=1,
            light=1
        )
        action = action.to_numpy()
        return action

class HeuristicAgent():
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
        if temp > 28.0:
            window = 1
        else:
            window = 0

        if temp < 22.0:
            heater = 1
        else:
            heater = 0
        humidity = state[2]
        if humidity < 80:
            vapor_supply = 1
        else:
            vapor_supply = 0

        if 5 < hour < 23:
            light_supply = 1
        else:
            light_supply = 0

        # scatterplot: daylight horizontaly and sun intensity
        # find the correlation between the

        action = GreenhouseAction(
            heater=heater,
            window=window,
            vapor_supply=vapor_supply,
            CO2_supply=CO2_supply,
            light=light_supply
        )
        action = action.to_numpy()
        return action
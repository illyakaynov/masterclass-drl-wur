import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# define function to calculate saturated vapor density
def saturated_vapor_density(temperature):
    # Calculate saturated vapor density mg/m3
    # For data see http://hyperphysics.phy-astr.gsu.edu/hbase/Kinetic/watvap.html
    # Saturated vapor density table has been fitted to 3rd polynome using Excel
    # Temperate: degrees
    # vapor density: gr/m3
    return (
        0.0006 * temperature ** 3
        - 0.0021 * temperature ** 2
        + 0.3322 * temperature
        + 5.8649
    )


def ppm_to_gpm3(CO2_ppm):
    # convert CO2 ppm to gram/m3
    return CO2_ppm * 44.01 / (24.45 * 1000)


def gpm3_to_ppm(CO2_gpm3):
    # convert CO2 gram/m3 to ppm
    return 24.45 * CO2_gpm3 * 1000 / 44.01


def rh_to_gpm3(humidity, temperature):
    # convert relative humidity to gram/m3
    return saturated_vapor_density(temperature) * humidity / 100.0


def gpm3_to_rh(vapor_density, temperature):
    return 100 * vapor_density / saturated_vapor_density(temperature)


class WeatherObservation:
    def __init__(
        self,
        temperature=17,  # degrees
        relative_humidity=40,  # %
        solar_power=0,  # W/m2
        CO2=350,  # ppm
    ):
        self.temperature = temperature
        self.relative_humidity = relative_humidity
        self.solar_power = solar_power
        self.CO2 = CO2

    def to_numpy(self):
        return np.array(
            [self.temperature, self.relative_humidity, self.solar_power, self.CO2]
        )

    @staticmethod
    def size():
        return 4

    @staticmethod
    def labels():
        return [
            "Temperature [C]",
            "Rel. Humidity [%]",
            "Solar power [W/m2]",
            "CO2 [ppm]",
        ]


class WeatherDefault:
    def __init__(
        self,
        temperature=17,  # degrees
        relative_humidity=40,  # %
        solar_power=0,  # W/m2
        CO2=350,
    ):  # ppm)
        self.obs = WeatherObservation(temperature, relative_humidity, solar_power, CO2)

    # default Weather model
    def reset(self):
        return self.obs

    def step(self):
        return self.obs


class Weather:
    # Weather model that uses weather data from csv file
    def __init__(self):
        # read data from cvs file
        # columns:
        # 'time', 'AbsHumOut', 'Iglob', 'PARout', 'Pyrgeo', 'RadSum', 'Rain',
        # 'Rhout', 'Tout', 'Winddir', 'Windsp'
        self.data = pd.read_csv("greenhouse_sim/data/meteo.csv").values[:, 1:].astype(np.float64)
        # correct nan values first two rows
        self.data[0, :] = self.data[2, :]
        self.data[1, :] = self.data[2, :]

    def reset(self):
        self.counter = 0
        obs = WeatherObservation(
            temperature=self.data[self.counter, 7],  # degrees
            relative_humidity=self.data[self.counter, 6],  # %
            solar_power=self.data[self.counter, 1],  # W/m2
            CO2=350,
        )
        return obs

    def step(self):
        self.counter += 1
        obs = WeatherObservation(
            temperature=self.data[self.counter, 7],  # degrees
            relative_humidity=self.data[self.counter, 6],  # %
            solar_power=self.data[self.counter, 1],  # W/m2
            CO2=350,
        )
        return obs


class PlantAction:
    def __init__(self, radiation=0, CO2=0, temperature=0):
        self.radiation = radiation  # W/m2
        self.CO2 = CO2  # g/m3 green house CO2
        self.temperature = temperature  # Celcius green house temperature

    def to_numpy(self):
        return np.array([self.radiation])

    @staticmethod
    def size():
        return 3

    @staticmethod
    def labels():
        return ["Radiation [W/m2]", "CO2 [g/m3]", "Temperature [Celcius]"]


class PlantObservation:
    def __init__(self, CO2_absorption_rate=0, CO2_total=0):
        self.CO2_absorption_rate = CO2_absorption_rate  # g/m2/s
        self.CO2_total = CO2_total

    def to_numpy(self):
        return np.array([self.CO2_absorption_rate, self.CO2_total])

    @staticmethod
    def size():
        return 2

    @staticmethod
    def labels():
        return ["CO2 absorption rate [g/m2/h]", "CO2 total [gram]"]


class Plant:
    def __init__(self):
        self.sample_time = 5

        # photosynthesis
        self.plant_CO2_generation = 0.1  # g/m2/h
        self.optimal_CO2_absorption = ppm_to_gpm3(800)  # g/m3

    def reset(self):
        self.total_CO2 = 0  # gram/m2
        return PlantObservation(0, 0)

    def step(self, action):
        # plant absorption
        if action.radiation > 0:
            CO2_absorption_rate = (
                -self.photosynthesis_rate(action) * self.optimal_CO2_absorption / 3600
            )
        else:
            CO2_absorption_rate = self.plant_CO2_generation / 3600

        self.total_CO2 += -CO2_absorption_rate * self.sample_time * 60
        return PlantObservation(CO2_absorption_rate, self.total_CO2)

    def photosynthesis_rate(self, action: PlantAction):
        temperature = np.clip(action.temperature, 0, 50)
        return (
            (1 - np.exp(-action.radiation / 100.0))
            * (1 - np.exp(-gpm3_to_ppm(action.CO2) / 400))
            * (1.0 - 0.0016 * (temperature - 25.0) ** 2)
        )


class GreenhouseAction:
    def __init__(self, heater=0, window=0, vapor_supply=0, CO2_supply=0, light=0):
        self.heater = heater  # heater on/off (1/0)
        self.window = window  # window open/closed (1/0)
        self.vapor_supply = vapor_supply  # vapor system on/off (1/0)
        self.CO2_supply = CO2_supply  # CO2 supply on/off (1/0)

    def to_numpy(self):
        return np.array([self.heater, self.window, self.vapor_supply, self.CO2_supply])

    @staticmethod
    def size():
        return 4

    @staticmethod
    def labels():
        return ["Heater on/off", "Window open/closes", "Vapor on/off", "CO2 on/off"]


class GreenhouseObservation:
    def __init__(self, time, temperature, humidity, CO2, weather, plant):
        self.time = time  # minutes  0 = 00:00 1 jan 2020
        self.temperature = temperature  # degrees
        self.humidity = humidity  # % relative humidity
        self.CO2 = CO2  # ppm
        self.weather = weather  # weather observation
        self.plant = plant

    def to_numpy(self):
        return np.concatenate(
            [
                np.array([self.time, self.temperature, self.humidity, self.CO2]),
                self.weather.to_numpy(),
                self.plant.to_numpy(),
            ]
        )

    @staticmethod
    def size():
        return 4 + WeatherObservation().size() + PlantObservation.size()

    @staticmethod
    def labels():
        return (
            ["Time [min]", "Temperature [C]", "Rel. Humidity [%]", "CO2 [ppm]"]
            + WeatherObservation().labels()
            + PlantObservation.labels()
        )


class Greenhouse:
    def __init__(self, weather_model=WeatherDefault(), plant_model=Plant()):
        # greenhouse dimensions
        self.area = 96  # m2
        self.height = 5  # m

        # green house specifications
        # heat capacity J/K/m2
        self.heat_capacity = 100000

        # heat loss coefficient W/K
        self.heat_loss_window_closed = 10
        self.heat_loss_window_open = 20

        # solar reflectance of glass
        self.reflectance = 0.5  # %

        # actuator specifications
        self.max_heating_capacity = 120  # W/m2
        self.max_vapor_capacity = 30  # g/m2/h
        self.max_CO2_capacity = 15  # g/m2/h
        self.max_ventilation_capacity = 5.0  # m3/h

        # vapor heat dissipation
        self.evaporation_coeff = 0.05  # m3/g
        self.evaporation_heat_dissipation = 10000  # W/g

        # simulation parameters
        self.sample_time = 5  # minutes

        # weather model
        self.weather_model = weather_model

        # plant model
        self.plant_model = plant_model

        # reward
        self.cost_heat = -0.01  # €/kW/m2
        self.cost_CO2 = -2000.0  # €/kg/m2
        self.cost_vapor = 0  # €/kg/m2
        self.plant_CO2 = 2000  # €/g

    def reset(self):
        self.time = 0  # minutes
        self.temp = 20  # degrees
        self.vapor_density = np.clip(5.0, 0, saturated_vapor_density(self.temp))  # g/m3
        self.CO2 = 0.6  # g/m3

        # total resource consumption
        self.total_heat = 0  # kW/m2
        self.total_CO2 = 0  # kg/m2
        self.total_vapor = 0  # kg/m2

        # reset weather model
        self.weather_obs = self.weather_model.reset()

        # reset plant model
        self.plant_obs = self.plant_model.reset()

        return GreenhouseObservation(
            self.time,
            self.temp,
            humidity=gpm3_to_rh(self.vapor_density, self.temp),
            CO2=gpm3_to_ppm(self.CO2),
            weather=self.weather_obs,
            plant=self.plant_obs,
        )

    def step(self, action: GreenhouseAction) -> GreenhouseObservation:
        reward = 0

        # simulate humidity
        vapor_sat = saturated_vapor_density(self.temp)
        if action.vapor_supply > 0:
            vapor_supply = (
                self.evaporation_coeff
                * (vapor_sat - self.vapor_density)
                * self.max_vapor_capacity
                / (self.height * 3600)
            )
            self.total_vapor += (
                self.max_vapor_capacity * self.sample_time * 60 / (3600 * 1000)
            )
            reward += (
                self.cost_vapor
                * self.max_vapor_capacity
                * self.sample_time
                * 60
                / (3600 * 1000)
            )
        else:
            vapor_supply = 0
        # ventilation
        alpha = np.clip(
            action.window * self.max_ventilation_capacity / self.height, 0, 1
        )
        vapor_ventilation = (
            alpha
            * (
                rh_to_gpm3(
                    self.weather_obs.relative_humidity, self.weather_obs.temperature
                )
                - self.vapor_density
            )
            / 3600
        )
        vapor_density_next = (
            self.vapor_density
            + (vapor_supply + vapor_ventilation) * self.sample_time * 60
        )

        # simulate temperature
        heat_supply_heater = action.heater * self.max_heating_capacity
        self.total_heat += heat_supply_heater * self.sample_time * 60 / 1000
        # reward += self.cost_heat * (heat_supply_heater * self.sample_time * 60 / 1000)
        heat_supply_solar = (1 - self.reflectance) * self.weather_obs.solar_power
        if action.window > 0:
            heat_loss_window = self.heat_loss_window_open * (
                self.temp - self.weather_obs.temperature
            )
        else:
            heat_loss_window = self.heat_loss_window_closed * (
                self.temp - self.weather_obs.temperature
            )
        heat_loss_vapor = self.evaporation_heat_dissipation * vapor_supply * self.height
        temp_next = (
            self.temp
            + (
                heat_supply_heater
                + heat_supply_solar
                - heat_loss_vapor
                - heat_loss_window
            )
            * self.sample_time
            * 60
            / self.heat_capacity
        )

        ### simulate CO2 concentration
        # supply
        CO2_supply = action.CO2_supply * self.max_CO2_capacity / (self.height * 3600)
        self.total_CO2 += CO2_supply * self.sample_time * 60 / 1000
        reward += self.cost_CO2 * CO2_supply * self.sample_time * 60 / 1000
        # ventilation
        alpha = np.clip(
            action.window * self.max_ventilation_capacity / self.height, 0, 1
        )
        CO2_ventilation = alpha * (ppm_to_gpm3(self.weather_obs.CO2) - self.CO2) / 3600
        # total
        CO2_next = (
            self.CO2
            + (CO2_supply + self.plant_obs.CO2_absorption_rate + CO2_ventilation)
            * self.sample_time
            * 60
        )
        if CO2_next < 0:
            CO2_next = 0

        # make observation
        obs = GreenhouseObservation(
            self.time,
            self.temp,
            humidity=gpm3_to_rh(self.vapor_density, self.temp),
            CO2=gpm3_to_ppm(self.CO2),
            weather=self.weather_obs,
            plant=self.plant_obs,
        )

        #
        # prepare next simulation step
        #
        # update weather
        self.weather_obs = self.weather_model.step()

        # update plant
        plant_action = PlantAction(
            radiation=self.weather_obs.solar_power, CO2=self.CO2, temperature=self.temp
        )
        self.plant_obs = self.plant_model.step(plant_action)
        reward += self.plant_CO2 * (-1 * self.plant_obs.CO2_absorption_rate)

        # update greenhouse variables
        self.temp = temp_next
        self.CO2 = CO2_next
        self.vapor_density = np.clip(
            vapor_density_next, 0, saturated_vapor_density(temp_next)
        )

        # update time
        self.time += self.sample_time

        # return observation
        return obs, reward


def plot_history(history):
    n_observations = GreenhouseObservation.size()
    n_actions = GreenhouseAction().size()
    n_vars = n_observations + n_actions - 1

    fig, axs = plt.subplots(n_vars, 1, figsize=(12, 2 * n_vars), sharex=True)

    for i in range(1, n_observations):
        ax = axs[i - 1]
        ax.plot(history[:, 0] / 60, history[:, i])
        ax.set_xlabel("Time [hours]")
        ax.set_ylabel(GreenhouseObservation.labels()[i])

    for i in range(0, n_actions):
        ax = axs[n_observations + i - 1]
        ax.plot(history[:, 0] / 60, history[:, n_observations + i])
        ax.set_xlabel("Time [hours]")
        ax.set_ylabel(GreenhouseAction.labels()[i])

    plt.tight_layout()


import gym
from gym import spaces

from itertools import product


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


if __name__ == "__main__":

    sim_max_minutes = 7 * 24 * 60 // 5

    history = np.zeros(
        (sim_max_minutes, GreenhouseObservation.size() + GreenhouseAction.size())
    )
    rewards = np.zeros(sim_max_minutes)

    # env = Greenhouse(weather_model=WeatherDefault(temperature=20.0))
    env = Greenhouse(weather_model=Weather())
    state = env.reset()

    for i in range(sim_max_minutes):
        hour = (env.time % (24 * 12)) / 12

        if (hour > 8) & (hour < 20) & (state.CO2 < 800):
            CO2_supply = 1
        else:
            CO2_supply = 0
        print(state.temperature)
        if state.temperature > 28.0:
            window = 1
        else:
            window = 0

        if state.temperature < 25.0:
            heater = 1
        else:
            heater = 0

        if state.humidity < 80:
            vapor_supply = 1
        else:
            vapor_supply = 0

        # calculate action
        action = GreenhouseAction(
            heater=heater,
            window=window,
            vapor_supply=vapor_supply,
            CO2_supply=CO2_supply,
        )
        # action = GreenhouseAction(heater=0, window=1, vapor_supply=1, CO2_supply=0)

        # do simulation step
        next_state, reward = env.step(action)

        # save history data
        history[i, :] = np.concatenate([state.to_numpy(), action.to_numpy()])
        rewards[i] = reward
        print(reward)

        # prepare next simulation state
        state = next_state

    plot_history(history)
    plt.show()


# %%

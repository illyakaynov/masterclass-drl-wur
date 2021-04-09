import matplotlib.pyplot as plt
import numpy as np
from greenhouse.convert_units import (
    gpm3_to_ppm,
    gpm3_to_rh,
    ppm_to_gpm3,
    rh_to_gpm3,
    saturated_vapor_density,
)
from greenhouse.plant import Plant, PlantAction, PlantObservation
from greenhouse.weather import Weather, WeatherDefault, WeatherObservation


class GreenhouseAction:
    def __init__(self, heater=0, window=0, vapor_supply=0, CO2_supply=0, light=0):
        self.heater = heater  # heater on/off (1/0)
        self.window = window  # window open/closed (1/0)
        self.vapor_supply = vapor_supply  # vapor system on/off (1/0)
        self.CO2_supply = CO2_supply  # CO2 supply on/off (1/0)
        self.light = light

    def to_numpy(self):
        return np.array(
            [self.heater, self.window, self.vapor_supply, self.CO2_supply, self.light]
        )

    @staticmethod
    def size():
        return 5

    @staticmethod
    def labels():
        return [
            "Heater on/off",
            "Window open/closes",
            "Vapor on/off",
            "CO2 on/off",
            "Light on/off",
        ]


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
            ["Time [min]", "GH Temperature [C]", "GH Rel. Humidity [%]", "GH CO2 [ppm]"]
            + WeatherObservation().labels()
            + PlantObservation.labels()
        )


class Greenhouse:
    def __init__(
        self,
        weather_model=None,
        plant_model=None,
        area=96,
        height=5,
        heat_capacity=100000,
        heat_loss_window_open=20,
        heat_loss_window_closed=10,
        reflectance=0.5,
        solar_power_to_par=4.6,
        max_heating_capacity=120,  # W/m2
        max_vapor_capacity=30,  # g/m2/h
        max_CO2_capacity=15,  # g/m2/h
        max_ventilation_capacity=5.0,  # m3/h
        max_light_capacity=187,  # umol/m2/s
        evaporation_coeff=0.05,  # m3/g
        evaporation_heat_dissipation=10000,  # W/g
        sample_time=5,  # minutes
        cost_heat=-0.01,  # €/kW/m2
        cost_CO2=-2000.0,  # €/kg/m2
        cost_vapor=1/1.08e5,  # €/kg/m2
        cost_light=-1 / 7.2e5,  # €/umol/m2
        plant_CO2=2000,  # €/g
        start_temp=20,  # degrees C
        start_CO2=0.6,  # g/m3
    ):

        # greenhouse dimensions
        self.area = area  # m2
        self.height = height  # m

        # green house specifications
        # heat capacity J/K/m2
        self.heat_capacity = heat_capacity

        # heat loss coefficient W/K
        self.heat_loss_window_closed = heat_loss_window_closed
        self.heat_loss_window_open = heat_loss_window_open

        # solar reflectance of glass
        self.reflectance = reflectance

        # solar radiation to PAR (photosynthesis active radiation)
        self.solar_power_to_par = solar_power_to_par

        # actuator specifications
        self.max_heating_capacity = max_heating_capacity  # W/m2
        self.max_vapor_capacity = max_vapor_capacity  # g/m2/h
        self.max_CO2_capacity = max_CO2_capacity  # g/m2/h
        self.max_ventilation_capacity = max_ventilation_capacity  # m3/h
        self.max_light_capacity = max_light_capacity  # umol/m2/s

        # vapor heat dissipation
        self.evaporation_coeff = evaporation_coeff  # m3/g
        self.evaporation_heat_dissipation = evaporation_heat_dissipation  # W/g

        # simulation parameters
        self.sample_time = sample_time  # minutes

        # weather model
        self.weather_model = weather_model or WeatherDefault()

        # plant model
        self.plant_model = plant_model or Plant()

        # reward
        self.cost_heat = cost_heat  # €/kW/m2
        self.cost_CO2 = cost_CO2  # €/kg/m2
        self.cost_vapor = cost_vapor  # €/kg/m2
        self.cost_light = cost_light  # €/umol/m2
        self.plant_CO2 = plant_CO2  # €/g

        self.start_temp = start_temp
        self.start_CO2 = start_CO2

    def reset(self):
        self.time = 0  # minutes
        self.temp = self.start_temp  # degrees
        self.vapor_density = np.clip(5.0, 0, saturated_vapor_density(self.temp))  # g/m3
        self.CO2 = self.start_CO2  # g/m3

        # total resource consumption
        self.total_heat = 0  # kW/m2
        self.total_CO2 = 0  # kg/m2
        self.total_vapor = 0  # kg/m2
        self.total_light = 0  # umol/m2

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
        reward_dict = {}
        reward = 0

        # simulate humidity
        vapor_sat = saturated_vapor_density(self.temp)
        cost_vapor = 0
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
            cost_vapor = (
                self.cost_vapor
                * self.max_vapor_capacity
                * self.sample_time
                * 60
                / (3600 * 1000)
            )
        else:
            vapor_supply = 0

        reward_dict['cost_vapor'] = cost_vapor
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
        reward_dict['cost_heat'] = self.cost_heat * (heat_supply_heater * self.sample_time * 60 / 1000)
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
        reward_dict['cost_CO2'] = self.cost_CO2 * CO2_supply * self.sample_time * 60 / 1000
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

        # update light model
        light_supply = action.light * self.max_light_capacity * self.sample_time * 60
        self.total_light += light_supply
        reward_dict['cost_light'] = self.cost_light * light_supply

        # make observation


        #
        # prepare next simulation step
        #
        # update weather
        self.weather_obs = self.weather_model.step()

        # update plant
        plant_action = PlantAction(
            radiation=self.solar_power_to_par * self.weather_obs.solar_power
            + action.light * self.max_light_capacity,
            CO2=self.CO2,
            temperature=self.temp,
        )
        self.plant_obs = self.plant_model.step(plant_action)
        reward_dict['CO2_absorbed'] = self.plant_CO2 * (-1 * self.plant_obs.CO2_absorption_rate)

        # update greenhouse variables
        self.temp = temp_next
        self.CO2 = CO2_next
        self.vapor_density = np.clip(
            vapor_density_next, 0, saturated_vapor_density(temp_next)
        )

        # update time
        self.time += self.sample_time

        # return observation
        reward = sum([v for v in reward_dict.values()])
        # print(reward_dict)

        obs = self.get_obs()

        return obs, reward, reward_dict

    def get_obs(self):
        obs = GreenhouseObservation(
            self.time,
            self.temp,
            humidity=gpm3_to_rh(self.vapor_density, self.temp),
            CO2=gpm3_to_ppm(self.CO2),
            weather=self.weather_obs,
            plant=self.plant_obs,
        )
        return obs



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

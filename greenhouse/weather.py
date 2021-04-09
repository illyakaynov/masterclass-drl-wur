import numpy as np
import pandas as pd


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
        attrs = [
            "Temperature [C]",
            "Rel. Humidity [%]",
            "Solar power [W/m2]",
            "CO2 [ppm]",
        ]
        attrs = ["Weather | {}".format(x) for x in attrs]
        return attrs


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
        self.data = (
            pd.read_csv("greenhouse/data/meteo.csv").values[:, 1:].astype(np.float64)
        )
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

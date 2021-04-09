import numpy as np

from greenhouse.convert_units import ppm_to_gpm3, gpm3_to_ppm


class PlantAction:
    def __init__(self, radiation=0, CO2=0, temperature=0):
        self.radiation = radiation  # W/m2
        self.CO2 = CO2  # g/m3 green house CO2
        self.temperature = temperature  # Celcius green house temperature

    def to_numpy(self):
        return np.array([self.radiation])

    @staticmethod
    def size():
        return 4

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
        attrs = ["CO2 absorption rate [g/m2/h]", "CO2 total [gram]"]
        attrs = ['Plant | {}'.format(attr) for attr in attrs]
        return attrs


class Plant:
    def __init__(self):
        self.sample_time = 5  # minutes

        # photosynthesis
        self.plant_CO2_generation = 0.1  # g/m2/h
        self.optimal_CO2_absorption = ppm_to_gpm3(800)  # g/m3

    def reset(self):
        self.total_CO2 = 0  # gram/m2
        self.time = 0
        return PlantObservation(0, 0)

    def step(self, action):
        hour = (self.time / 60) % 24

        # simple equation to model fact that plant needs rest
        if (hour > 5.0) and (hour < 23.0):
            CO2_absorption_rate = (
                -self.photosynthesis_rate(action) * self.optimal_CO2_absorption / 3600
            )
        else:
            CO2_absorption_rate = self.plant_CO2_generation / 3600

        # update total CO2
        self.total_CO2 += -CO2_absorption_rate * self.sample_time * 60

        # update time
        self.time += self.sample_time

        return PlantObservation(CO2_absorption_rate, self.total_CO2)

    def photosynthesis_rate(self, action: PlantAction):
        return (
            (1.0 - np.exp(-action.radiation / (500.0)))
            * (1.0 - np.exp(-gpm3_to_ppm(action.CO2) / 400))
            * (1.0 - 0.0016 * (np.clip(action.temperature, 0.0, 50.0) - 25.0) ** 2)
        )

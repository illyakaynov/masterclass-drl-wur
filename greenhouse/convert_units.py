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
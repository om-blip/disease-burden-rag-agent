def calculate_yld(incidence, duration, dw):
    return incidence * duration * dw


def calculate_yll(deaths, life_expectancy):
    return deaths * life_expectancy


def estimate_duration():
    return "Duration can be estimated from survival literature or registry studies."
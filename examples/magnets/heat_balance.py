import numpy as np
from scipy.integrate import odeint


def heat_balance_model(T, t, I, C_components, R0, alpha, initial_temperature):
    """
    Calculate the derivative of temperature (dT/dt) for the heat balance problem.

    Parameters:
        T : float
            The current temperature in Celsius.
        t : float
            The current time in seconds.
        I : float
            The current flowing through the conductor in Amperes.
        C_components : list of floats
            List of heat capacities of conductor components in Joules per Kelvin.
        R0 : float
            Initial electrical resistance at reference temperature in Ohms.
        alpha : float
            Temperature coefficient of resistance (per degree Celsius).
        initial_temperature : float
            The initial temperature in Celsius.

    Returns:
        dTdt : float
            The derivative of temperature with respect to time (dT/dt).
    """
    # Calculate the rate of heat generation (Joule dissipation)
    Q_gen = I**2 * resistance_function(T, R0, alpha, initial_temperature)

    # Calculate the rate of heat absorption by conductor components
    Q_abs = sum(C_i * (T - initial_temperature) for C_i in C_components)

    # Calculate the derivative of temperature with respect to time (dT/dt)
    dTdt = Q_gen / Q_abs

    return dTdt


def resistance_function(T, R0, alpha, initial_temperature):
    """
    Calculate the electrical resistance as a function of temperature.

    Parameters:
        T : float
            The current temperature in Celsius.
        R0 : float
            Initial electrical resistance at reference temperature in Ohms.
        alpha : float
            Temperature coefficient of resistance (per degree Celsius).
        initial_temperature : float
            The initial temperature in Celsius.

    Returns:
        float
            The electrical resistance at the given temperature.
    """
    return R0 * (1 + alpha * (T - initial_temperature))


# Example usage:
# Example usage with hypothetical input values:
I = 2.0
C_components = [100, 150, 200]
R0 = 10.0
alpha = 0.03
initial_temperature = 25.0
time_steps = np.linspace(0, 10, 1000)

# Solve the ODE using odeint
solution = odeint(
    heat_balance_model,
    initial_temperature,
    time_steps,
    args=(I, C_components, R0, alpha, initial_temperature),
)

final_temperature = solution[-1]
print(f"Final temperature: {final_temperature} Celsius")

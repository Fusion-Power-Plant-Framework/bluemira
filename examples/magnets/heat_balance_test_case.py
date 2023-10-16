import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


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
    Q_abs = sum(C_i for C_i in C_components)

    # Calculate the derivative of temperature with respect to time (dT/dt)
    dTdt = Q_gen / Q_abs

    return dTdt


def temperature_evolution(I, C_components, R0, alpha, initial_temperature, time_steps):
    # Solve the ODE using odeint
    solution = odeint(
        heat_balance_model,
        initial_temperature,
        time_steps,
        args=(I, C_components, R0, alpha, initial_temperature),
    )
    return solution


# Example usage:
# Example usage with hypothetical input values:
I = 2.0
C_components = [100, 150, 200]
R0 = 10.0
alpha = 0.04
initial_temperature = 25.0
target_temperature = 250.0
time_steps = np.linspace(0, 1000, 1000)


for alpha in np.linspace(0, 0.3, 10):
    # Solve the ODE using odeint
    solution = temperature_evolution(
        I, C_components, R0, alpha, initial_temperature, time_steps
    )

    final_temperature = solution[-1]
    print(f"Final temperature: {final_temperature} Celsius")

    # plt.plot(time_steps, solution.T[0])
    # plt.show()

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import odeint
    from scipy.optimize import minimize

    # Define the function that computes the final temperature for a given alpha
    def final_temperature_difference(
        alpha, I, C_components, R0, initial_temperature, target_temperature
    ):
        time_steps = np.linspace(0, 1000, 1000)

        def heat_balance_residual():
            solution = temperature_evolution(
                I, C_components, R0, alpha, initial_temperature, time_steps
            )
            final_T = solution[-1]
            return abs(final_T - target_temperature)

        return (
            heat_balance_residual()
        )  # Compute the difference at the initial temperature

    # Use scipy.optimize.root_scalar to find alpha
    result = minimize(
        final_temperature_difference,
        alpha,
        args=(I, C_components, R0, initial_temperature, target_temperature),
        bounds=[(0.001, 0.03)],  # Adjust the bracket for alpha
    )

    if result.success:
        optimal_alpha = result.x[0]

        # Solve the ODE again with the optimized alpha and target temperature as initial condition
        # Solve the ODE using odeint
        solution = temperature_evolution(
            I, C_components, R0, optimal_alpha, initial_temperature, time_steps
        )

        final_temperature = solution[-1]

        # plt.plot(time_steps, solution.T[0])
        # plt.show()

        print(f"Optimal alpha: {optimal_alpha}")
        print(f"Final temperature with optimal alpha: {final_temperature[0]} Celsius")
    else:
        print(
            "Optimization did not converge. Check your input parameters or initial bracket."
        )

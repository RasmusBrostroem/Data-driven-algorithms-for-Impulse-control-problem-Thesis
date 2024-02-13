import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union

# Define the drift function and diffusion coefficient
def b(x: float, t: float) -> float:
    # return the drift function evaluated at x and t
    return -x/2

def sigma(x: float, t: float) -> float:
    # return the diffusion coefficient evaluated at x and t
    return 1

def EulerMaruymaMethod(b: Callable[[float, float], float],
                       sigma: Callable[[float, float], float],
                       T: float,
                       dt: float,
                       x0: float) -> Union[np.array, np.array]:
    """Simmulating the Itô diffusion process using Euler Maruyma Method
    See https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method

    Args:
        b (Callable[[float, float], float]): The drift function
        sigma (Callable[[float, float], float]): The diffusion coefficient
        T (float): Number of time the process should run
        dt (float): The increment of time before simulating new value
        x0 (float): Initial value of process

    Returns:
        Union[np.array, np.array]: 
            x-values: the simulated values of the process
            t-values: the time steps, where simulated values were generated
    """
    N = int(T/dt) # number of time steps
    x = np.zeros(N+1)
    t = np.zeros(N+1)
    x[0] = x0
    t[0] = 0.0

    for n in range(N):
        t[n+1] = t[n] + dt
        z = np.random.normal(loc=0.0, scale=np.sqrt(dt))
        x[n+1] = x[n] + b(x[n], t[n])*dt + sigma(x[n], t[n])*z

    return x, t

x, t = EulerMaruymaMethod(b, sigma, 100, 0.01, 0)

# Plot the simulated values
plt.plot(t, x)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
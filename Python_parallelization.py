import numpy as np
import time

def calculate_pi(N):
    start_time = time.time()  # Inicia el contador de tiempo
    delta_x = 1 / N
    sum_area = 0
    for i in range(N):
        xi = i * delta_x
        fi = np.sqrt(1 - xi**2)
        sum_area += fi * delta_x
    pi = 4 * sum_area
    end_time = time.time()  # Finaliza el contador de tiempo
    print(f"Time taken: {end_time - start_time} seconds")  # Imprime el tiempo transcurrido
    return pi

# Example usage:
N = 1000000
approx_pi = calculate_pi(N)
print("Approximation of pi:", approx_pi)

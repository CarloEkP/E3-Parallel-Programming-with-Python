import numpy as np
import time
from multiprocessing import Pool

def worker(args):
    start, end, N = args
    delta_x = 1 / N
    sum_area = 0
    for i in range(start, end):
        xi = i * delta_x
        fi = np.sqrt(1 - xi**2)
        sum_area += fi * delta_x
    return sum_area

def calculate_pi_parallel(N, num_processes):
    start_time = time.time()  # Inicia el contador de tiempo
    pool = Pool(processes=num_processes)
    chunk_size = N // num_processes
    tasks = [(i * chunk_size, (i + 1) * chunk_size, N) for i in range(num_processes)]
    results = pool.map(worker, tasks)
    pool.close()
    pool.join()
    pi = 4 * sum(results)
    end_time = time.time()  # Finaliza el contador de tiempo
    print(f"Time taken: {end_time - start_time} seconds")  # Imprime el tiempo transcurrido
    return pi

# Example usage:
N = 1000000
num_processes = 4
approx_pi_parallel = calculate_pi_parallel(N, num_processes)
print("Parallel approximation of pi:", approx_pi_parallel)


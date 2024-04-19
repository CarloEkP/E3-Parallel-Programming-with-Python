from mpi4py import MPI
import numpy as np
import time

def calculate_pi_distributed(N):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.time() if rank == 0 else None  # Inicia el contador de tiempo solo en el proceso raíz

    delta_x = 1 / N
    local_n = N // size
    local_start = rank * local_n
    local_end = (rank + 1) * local_n if rank != size - 1 else N

    local_sum = 0
    for i in range(local_start, local_end):
        xi = i * delta_x
        fi = np.sqrt(1 - xi**2)
        local_sum += fi * delta_x

    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        pi = 4 * global_sum
        end_time = time.time()  # Finaliza el contador de tiempo en el proceso raíz
        print(f"Time taken: {end_time - start_time} seconds")  # Imprime el tiempo transcurrido
        return pi

# To run this, execute it using an MPI command, e.g., mpirun -np 4 python3 script.py
if __name__ == "__main__":
    N = 1000000
    pi = calculate_pi_distributed(N)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Distributed approximation of pi:", pi)

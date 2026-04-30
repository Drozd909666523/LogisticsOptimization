import numpy as np


def build_transportation_matrices(supply, demand, cost_matrix):
    """
    Создает матрицы A, b и вектор c для транспортной задачи.
    supply: список запасов на складах (длина N)
    demand: список потребностей магазинов (длина M)
    cost_matrix: матрица стоимости перевозки NxM
    """
    N = len(supply)
    M = len(demand)
    num_vars = N * M

    c = np.array(cost_matrix).flatten()

    A = []
    b = []

    for i in range(N):
        row = np.zeros(num_vars)
        row[i * M : (i + 1) * M] = 1
        A.append(row)
        b.append(supply[i])

    for j in range(M):
        row = np.zeros(num_vars)
        for i in range(N):
            row[i * M + j] = 1
        A.append(row)
        b.append(demand[j])

    A = np.array(A)
    b = np.array(b)

    A = A[:-1]
    b = b[:-1]

    return c, A, b

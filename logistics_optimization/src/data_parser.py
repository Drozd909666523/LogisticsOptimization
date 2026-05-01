import json

import numpy as np


def haversine_distance(coord1, coord2):
    """
    Вычисляет расстояние в километрах между двумя точками на Земле.
    Координаты передаются в формате (широта, долгота).
    """
    R = 6371.0
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def load_shops_from_overpass(filepath):
    """Читает JSON от Overpass и возвращает список координат (lat, lon)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    shops = []
    for element in data.get("elements", []):
        if element["type"] == "node":
            shops.append((element["lat"], element["lon"]))
    return shops


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

    return c, A[:-1], b[:-1]

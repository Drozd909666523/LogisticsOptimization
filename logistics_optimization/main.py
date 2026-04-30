import numpy as np

from src.data_parser import (
    build_transportation_matrices,
    haversine_distance,
    load_shops_from_overpass,
)
from src.optimizer import interior_point


def main():
    shops_coords = load_shops_from_overpass("data/raw_data.json")
    num_shops = len(shops_coords)
    print(f"Найдено магазинов: {num_shops}")

    warehouses_coords = [(60.066, 30.333), (59.810, 30.380), (59.950, 30.560)]
    num_warehouses = len(warehouses_coords)

    cost_matrix = np.zeros((num_warehouses, num_shops))
    for i, w in enumerate(warehouses_coords):
        for j, s in enumerate(shops_coords):
            cost_matrix[i, j] = haversine_distance(w, s)

    np.random.seed(42)
    demand = np.random.randint(10, 50, size=num_shops)
    total_goods = np.sum(demand)

    supply = np.array(
        [total_goods // 3, total_goods // 3, total_goods - 2 * (total_goods // 3)]
    )

    print(f"Общий спрос (паллет): {total_goods}")
    print(f"Запасы на складах: {supply}")

    c, A_eq, b_eq = build_transportation_matrices(supply, demand, cost_matrix)

    print("\nЗапуск метода внутренней точки...")

    num_routes = num_warehouses * num_shops
    avg_flow = total_goods / num_routes

    x0 = np.ones(num_routes) * avg_flow

    result = interior_point(c, A_eq, b_eq, x0=x0, gamma=0.9, max_iter=1000)

    if result["success"]:
        print(
            f"Успех! Минимальные затраты на логистику (тонно-километры): {result['fun']:.2f}"
        )

        optimal_routes = result["x"].reshape((num_warehouses, num_shops))

        print("\nПлан поставок со склада 'Парнас':")
        for j in range(num_shops):
            if optimal_routes[0, j] > 0.5:
                print(f"  В магазин №{j + 1}: {optimal_routes[0, j]:.1f} паллет")
    else:
        print("Оптимизация не удалась:", result["message"])


if __name__ == "__main__":
    main()

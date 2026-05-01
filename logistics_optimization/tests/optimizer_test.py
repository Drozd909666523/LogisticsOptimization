import unittest

import numpy as np
from scipy.optimize import linprog

from logistics_optimization.src.optimizer import interior_point, central_path


class TestInteriorPointOptimizer(unittest.TestCase):
    def test_simple_lp_problem(self):
        """
        Тестируем задачу линейного программирования:
        Минимизировать: z = -3x1 - 5x2
        Ограничения (приведенные к равенствам с балансировочными переменными x3, x4, x5):
        x1 + x3 = 4
        2x2 + x4 = 12
        3x1 + 2x2 + x5 = 18
        """
        # Вектор стоимостей (целевая функция)
        c = [-3, -5, 0, 0, 0]

        # Матрица ограничений A_eq
        A_eq = [[1, 0, 1, 0, 0], [0, 2, 0, 1, 0], [3, 2, 0, 0, 1]]

        # Вектор правых частей b_eq
        b_eq = [4, 12, 18]

        expected_result = linprog(
            c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs"
        )

        result = interior_point(c, A_eq, b_eq)

        self.assertTrue(expected_result.success, "Scipy не смог решить задачу")
        self.assertTrue(result["success"], "Тестируемый метод не смог решить задачу")

        self.assertAlmostEqual(
            expected_result.fun,
            result["fun"],
            places=4,
            msg="Значения целевой функции не совпадают",
        )

        np.testing.assert_allclose(
            expected_result.x,
            result["x"],
            atol=1e-4,
            err_msg="Векторы решений не совпадают",
        )


    def test_central_path_problem(self):
        """
        Тестируем метод центрального пути на той же базовой задаче.
        """
        c = [-3, -5, 0, 0, 0]

        A_eq = [
            [1, 0, 1, 0, 0],
            [0, 2, 0, 1, 0],
            [3, 2, 0, 0, 1]
        ]
        b_eq = [4, 12, 18]

        x0 = np.ones(5)

        res = central_path(c, A_eq, b_eq, x0=x0, mu0=10.0)

        self.assertTrue(res["success"], "Алгоритм центрального пути не сошелся")

        self.assertAlmostEqual(-36.0, res["fun"], places=3,
                               msg=f"Неверный ответ! Ожидалось -36.0, получено {res['fun']}")


if __name__ == "__main__":
    unittest.main()

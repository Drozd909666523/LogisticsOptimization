import numpy as np


def custom_interior_point(c, A_eq, b_eq, x0=None, gamma=0.95, tol=1e-6, max_iter=200):
    """
    Реализация прямого метода внутренней точки (алгоритм Дикина).
    Решает задачу: min c^T x, при A_eq x = b_eq, x >= 0.
    """
    c = np.array(c, dtype=float)
    A = np.array(A_eq, dtype=float)
    b = np.array(b_eq, dtype=float)

    m, n = A.shape

    if x0 is None:
        x = np.ones(n)
    else:
        x = np.array(x0, dtype=float)

    for k in range(max_iter):
        r = b - A @ x

        if np.linalg.norm(r) < tol and k > 0:
            break

        D = np.diag(x**2)

        LHS = A @ D @ A.T
        RHS = r + A @ D @ c

        try:
            u = np.linalg.solve(LHS, RHS)
        except np.linalg.LinAlgError:
            print(f"Матрица стала сингулярной на итерации {k}")
            break

        g = c - A.T @ u

        s = -D @ g

        negative_s = s < 0
        if np.any(negative_s):
            lambda_tilde = gamma * np.min(-x[negative_s] / s[negative_s])
        else:
            lambda_tilde = 1.0

        if np.linalg.norm(r) > tol:
            step = min(1.0, lambda_tilde)
        else:
            step = lambda_tilde

        x_new = x + step * s

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    return {
        "x": x,
        "fun": c @ x,
        "nit": k,
        "success": k < max_iter - 1,
        "message": "Optimization terminated successfully."
        if k < max_iter - 1
        else "Maximum iterations reached.",
    }

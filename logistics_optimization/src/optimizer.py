import numpy as np


def interior_point(c, A_eq, b_eq, x0=None, gamma=0.95, tol=1e-6, max_iter=200):
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

        D = np.diag(np.clip(x**2, 1e-12, None))

        LHS = A @ D @ A.T
        RHS = r + A @ D @ c

        u = np.linalg.solve(LHS, RHS)

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


def central_path(c, A_eq, b_eq, x0, mu0=10.0, tol=1e-6, max_iter=200):
    c = np.array(c, dtype=float)
    A = np.array(A_eq, dtype=float)
    b = np.array(b_eq, dtype=float)
    x = np.array(x0, dtype=float)

    m, n = A.shape
    mu = mu0
    gamma = 0.95

    rho_n = 0.5 * (np.sqrt(n) - 1) / (n - 0.5)

    for k in range(max_iter):
        r = b - A @ x
        D = np.diag(np.clip(x ** 2, 1e-12, None))
        LHS = A @ D @ A.T

        if np.linalg.norm(r) > tol:
            RHS = r + A @ D @ c
            try:
                u = np.linalg.solve(LHS, RHS)
            except np.linalg.LinAlgError:
                u = np.linalg.lstsq(LHS, RHS, rcond=None)[0]

            g = c - A.T @ u
            s = -D @ g
            mu_new = mu

        else:
            r = np.zeros_like(r)
            RHS_0 = A @ D @ c
            RHS_mu = (A @ D @ c) - (mu * b)

            try:
                theta_0 = np.linalg.solve(LHS, RHS_0)
                theta_mu = np.linalg.solve(LHS, RHS_mu)
            except np.linalg.LinAlgError:
                theta_0 = np.linalg.lstsq(LHS, RHS_0, rcond=None)[0]
                theta_mu = np.linalg.lstsq(LHS, RHS_mu, rcond=None)[0]

            mu_new = (1 - rho_n) * mu
            u = theta_0 + (mu_new / mu) * (theta_mu - theta_0)
            g = c - A.T @ u

            s = (1 / mu_new) * x * (mu_new - x * g)

        negative_s = s < 0
        if np.any(negative_s):
            step = min(1.0, gamma * np.min(-x[negative_s] / s[negative_s]))
        else:
            step = 1.0

        x_new = x + step * s

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new
        mu = mu_new

    return {
        "x": x,
        "fun": c @ x,
        "nit": k,
        "success": k < max_iter - 1,
        "name": "Central Path"
    }

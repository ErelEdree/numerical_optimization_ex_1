import numpy as np

def quadratic_identity(x, is_hessian_needed=True):
    Q = np.array([[1, 0], [0, 1]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if is_hessian_needed else None
    return f, g, h

def quadratic_ellipse(x, is_hessian_needed=True):
    Q = np.array([[1, 0], [0, 100]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if is_hessian_needed else None
    return f, g, h

def quadratic_rotated(x, is_hessian_needed=True):
    theta = np.pi / 6  # 30 degrees
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    D = np.diag([100, 1])
    Q = R.T @ D @ R
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if is_hessian_needed else None
    return f, g, h

def rosenbrock(x, is_hessian_needed=True):
    x1, x2 = x[0], x[1]
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    g = np.array([
        -400 * x1 * (x2 - x1**2) - 2 * (1 - x1),
        200 * (x2 - x1**2)
    ])
    h = np.array([
        [1200 * x1**2 - 400 * x2 + 2, -400 * x1],
        [-400 * x1, 200]
    ]) if is_hessian_needed else None
    return f, g, h

def linear_function(x, is_hessian_needed=True):
    a = np.array([1.0, -2.0])  # can be any nonzero vector
    f = a.T @ x
    g = a
    h = np.zeros((2, 2)) if is_hessian_needed else None  # second derivative is zero for linear functions
    return f, g, h

def exponential_triangle(x, is_hessian_needed=True):
    x1, x2 = x[0], x[1]
    t1 = np.exp(x1 + 3 * x2 - 0.1)
    t2 = np.exp(x1 - 3 * x2 - 0.1)
    t3 = np.exp(-x1 - 0.1)
    f = t1 + t2 + t3

    g = np.array([
        t1 + t2 - t3,
        3 * t1 - 3 * t2
    ])

    if is_hessian_needed:
        h11 = t1 + t2 + t3
        h12 = 3 * t1 - 3 * t2
        h22 = 9 * t1 + 9 * t2
        h = np.array([
            [h11, h12],
            [h12, h22]
        ])
    else:
        h = None

    return f, g, h
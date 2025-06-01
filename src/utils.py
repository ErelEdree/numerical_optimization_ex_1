import numpy as np
import matplotlib.pyplot as plt

def plot_contours_with_paths(f, path_gd, path_nt, title="", filename=None, save=False, show=False):
    # Combine both paths to determine the range
    all_points = np.vstack([path_gd, path_nt])
    
    # Calculate min and max with padding
    padding = 0.2# 20% padding
    x_min = np.min(all_points[:, 0]) * (1 + padding)
    if x_min == 0:
        x_min = -1
    x_max = np.max(all_points[:, 0]) * (1 + padding)
    if x_max == 0:
        x_max = 1
    y_min = np.min(all_points[:, 1]) * (1 + padding)
    if y_min == 0:
        y_min = -1
    y_max = np.max(all_points[:, 1]) * (1 + padding)
    if y_max == 0:
        y_max = 1
    
    # Ensure we have some minimum range
    min_range = 0.1
    if x_max - x_min < min_range:
        center = (x_max + x_min) / 2
        x_min = center - min_range/2
        x_max = center + min_range/2
    if y_max - y_min < min_range:
        center = (y_max + y_min) / 2
        y_min = center - min_range/2
        y_max = center + min_range/2

    X, Y = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j], _, _ = f(x)

    plt.figure(figsize=(8, 6))
    CS = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.clabel(CS, inline=1, fontsize=8)
    plt.plot(path_gd[:, 0], path_gd[:, 1], 'r.-', label="Gradient Descent")
    plt.plot(path_nt[:, 0], path_nt[:, 1], 'b:.', label="Newton's Method")
    plt.title(title)
    plt.legend()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    if filename and save:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()

def plot_function_values(fvals_gd, fvals_nt, title="", filename=None, save=False, show=False):
    plt.figure(figsize=(8, 6))
    plt.plot(fvals_gd, 'r-', label="Gradient Descent")
    plt.plot(fvals_nt, 'b:.', label="Newton's Method")
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(title)
    plt.legend()
    if filename and save:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
import numpy as np
import matplotlib.pyplot as plt

def plot_contours_with_paths(f, path_gd, path_nt, title="", filename=None, save=False, show=False):
    # Combine all path points
    all_points = np.vstack([path_gd, path_nt])

    # Calculate full range from actual path data
    x_vals = all_points[:, 0]
    y_vals = all_points[:, 1]

    x_min = np.min(x_vals)
    x_max = np.max(x_vals)
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)

    # Add outward padding
    padding = 0.2
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range

    # Ensure minimum plot range
    min_range = 0.1
    if x_max - x_min < min_range:
        center = (x_max + x_min) / 2
        x_min = center - min_range / 2
        x_max = center + min_range / 2
    if y_max - y_min < min_range:
        center = (y_max + y_min) / 2
        y_min = center - min_range / 2
        y_max = center + min_range / 2

    # Create grid
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    points = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Evaluate function
    Z_vals = np.array([f(x)[0] for x in points])
    Z = Z_vals.reshape(X.shape)

    # Clip Z to compress extremes
    z1, z99 = np.percentile(Z, 1), np.percentile(Z, 99)
    Z = np.clip(Z, z1, z99)

    # Plot
    plt.figure(figsize=(8, 6))
    CS = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.clabel(CS, inline=1, fontsize=8)
    
    # Plot paths with start and end markers
    plt.plot(path_gd[:, 0], path_gd[:, 1], 'r.-', label="Gradient Descent")
    plt.plot(path_nt[:, 0], path_nt[:, 1], 'b:.', label="Newton's Method")
    
    # Mark start points
    plt.plot(path_gd[0, 0], path_gd[0, 1], 'ro', markersize=6, label='GD Start')
    plt.plot(path_nt[0, 0], path_nt[0, 1], 'bo', markersize=6, label='NT Start')
    
    # Mark end points
    plt.plot(path_gd[-1, 0], path_gd[-1, 1], 'r*', markersize=8, label='GD End')
    plt.plot(path_nt[-1, 0], path_nt[-1, 1], 'b*', markersize=8, label='NT End')
    
    plt.title(title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)

    if filename and save:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_function_values(fvals_gd, fvals_nt, title="", filename=None, save=False, show=False):
    plt.figure(figsize=(8, 6))
    plt.plot(fvals_gd, 'r-', label="Gradient Descent")
    plt.plot(fvals_nt, 'b:.', label="Newton's Method")
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if filename and save:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
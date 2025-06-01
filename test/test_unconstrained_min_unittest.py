import unittest
import numpy as np
from test.examples import (
    quadratic_identity,
    quadratic_ellipse,
    quadratic_rotated,
    rosenbrock,
    linear_function,
    exponential_triangle
)
from src.unconstrained_min import UnconstrainedMin
from src.utils import plot_contours_with_paths, plot_function_values

class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        self.functions = {
            "Quadratic Identity": quadratic_identity,
            "Quadratic Ellipse": quadratic_ellipse,
            "Quadratic Rotated": quadratic_rotated,
            "Rosenbrock": rosenbrock,
            "Linear Function": linear_function,
            "Exponential Triangle": exponential_triangle
        }
        
        self.initial_points = {
            "Rosenbrock": np.array([-1.0, 2.0]),
            "default": np.array([1.0, 1.0])
        }

    def test_minimization_methods(self):
        for name, func in self.functions.items():
            with self.subTest(function_name=name):
                x0 = self.initial_points["Rosenbrock"] if name == "Rosenbrock" else self.initial_points["default"]
                
                # Initialize minimizers
                gd = UnconstrainedMin(
                    func, 
                    x0, 
                    obj_tol=1e-12, 
                    param_tol=1e-8, 
                    max_iter=(10000 if name == "Rosenbrock" else 100), 
                    method="gradient_descent"
                )
                nt = UnconstrainedMin(
                    func, 
                    x0, 
                    obj_tol=1e-12, 
                    param_tol=1e-8, 
                    max_iter=100, 
                    method="newton_method"
                )

                # Run minimization
                result_gd = gd.minimize()
                result_nt = nt.minimize()


                # Plot results
                plot_contours_with_paths(
                    func,
                    result_gd["path"],
                    result_nt["path"],
                    title=f"Contours + Paths: {name}",
                    filename=f"{name.replace(' ', '_').lower()}_paths.png",
                    save=True,
                )

                plot_function_values(
                    result_gd["f_values"],
                    result_nt["f_values"],
                    title=f"Function Values: {name}",
                    filename=f"{name.replace(' ', '_').lower()}_values.png",
                    save=True,
                )


if __name__ == '__main__':
    unittest.main() 
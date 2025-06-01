import numpy as np

class UnconstrainedMin:
    def __init__(self, f, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method="gradient_descent"):
        #assumption - f(x) returns a tuple of (f_val, grad, hess)
        self.f = f
        self.x0 = x0
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.method = method

    def minimize(self):
        if self.method == "gradient_descent":
            return self.gradient_descent()
        elif self.method == "newton_method":
            return self.newton_method()
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def gradient_descent(self):
        x = self.x0.copy()
        path = [x.copy()]
        f_val, grad, _ = self.f(x, is_hessian_needed=False)
        f_values = [f_val]

        for i in range(self.max_iter):
            p = -grad
            alpha = self.backtracking_line_search(x, p, grad)
            x_new = x + alpha * p
            f_new, grad, _ = self.f(x_new, is_hessian_needed=False)

            # print(f"Iter {i}: x = {x_new}, f(x) = {f_new}")
            
            if abs(f_new - f_values[-1]) < self.obj_tol or np.linalg.norm(x_new - x) < self.param_tol:
                path.append(x_new.copy())
                f_values.append(f_new)
                print(f"function: {self.f.__name__}, method: gradient_descent, Iter {i}: x = {x_new}, f(x) = {f_new}, success = True")
                return {
                    'x': x_new,
                    'success': True,
                    'iterations': i + 1,
                    'path': np.array(path),
                    'f_values': np.array(f_values)
                }

            x = x_new
            path.append(x.copy())
            f_values.append(f_new)
        print(f"function: {self.f.__name__}, method: gradient_descent, Iter {self.max_iter}: x = {x}, f(x) = {f_values[-1]}, success = False")
        return {
            'x': x,
            'success': False,
            'iterations': self.max_iter,
            'path': np.array(path),
            'f_values': np.array(f_values)
        }

    def newton_method(self):
        x = self.x0.copy()
        path = [x.copy()]
        f_val, grad, hess = self.f(x, is_hessian_needed=True)
        f_values = [f_val]

        for i in range(self.max_iter):
            try:
                p = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # print(f"Iter {i}: Hessian is not invertible.")
                break

            alpha = self.backtracking_line_search(x, p, grad)
            x_new = x + alpha * p
            f_new, grad, hess = self.f(x_new, is_hessian_needed=True)

            # print(f"Iter {i}: x = {x_new}, f(x) = {f_new}")

            if abs(f_new - f_values[-1]) < self.obj_tol or np.linalg.norm(x_new - x) < self.param_tol:
                path.append(x_new.copy())
                f_values.append(f_new)
                print(f"function: {self.f.__name__}, method: newton_method, Iter {i}: x = {x_new}, f(x) = {f_new}, success = True")
                return {
                    'x': x_new,
                    'success': True,
                    'iterations': i + 1,
                    'path': np.array(path),
                    'f_values': np.array(f_values)
                }

            x = x_new
            path.append(x.copy())
            f_values.append(f_new)
        print(f"function: {self.f.__name__}, method: newton_method, Iter {self.max_iter}: x = {x}, f(x) = {f_values[-1]}, success = False")
        return {
            'x': x,
            'success': False,
            'iterations': self.max_iter,
            'path': np.array(path),
            'f_values': np.array(f_values)
        }

    def backtracking_line_search(self, x, p, grad, alpha=1.0, rho=0.5, c=0.01):
        f_x, _, _ = self.f(x, is_hessian_needed=False)
        while True:
            x_new = x + alpha * p
            f_new, _, _ = self.f(x_new, is_hessian_needed=False)
            if f_new <= f_x + c * alpha * np.dot(grad, p):
                break
            alpha *= rho
        return alpha
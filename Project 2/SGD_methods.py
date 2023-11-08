import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from tqdm import tqdm


def derivative_OLS(y: np.array, X: np.array, beta: np.array, lam=None):
    """Computes the derivate with respect to beta of the OLS cost function."""
    n = X.shape[0]
    return -2.0 / n * np.dot(X.T, y - np.dot(X, beta))


def CostOLS(y: np.array, X: np.array, beta: np.array, lam=None) -> np.array:
    """Cost function using OLS regression

    Args:
        y (np.array): True value
        X (np.array): Design matrix
        beta (np.array): Beta parameters
        lam (_type_, optional): Dummy variable so it is easier to write code for Ridge regression. Defaults to None.

    Returns:
        np.array: _description_
    """
    n = X.shape[0]
    return np.sum((y - X @ beta) ** 2) / n


def CostRidge(y: np.array, X: np.array, beta: np.array, lam: float):
    """Computes the cost function for Ridge regression.

    Args:
        y (np.array):       Array of y-values
        X (np.array):       Design matrix.
        beta (np.array):    Array of beta coefficients.
        lam (float):        Regularisation constant.

    Returns:
        np.array: Cost function for Ridge regression
    """
    return np.sum((y - X @ beta) ** 2) + lam * beta.T @ beta


class SGD:
    """Class for performing stochastic gradient ascent (SGD) for OLS or Ridge regression"""

    def __init__(
        self,
        X: np.array,
        y: np.array,
        derivative,
        lam: float = None,
        reg_param: float = 0,
    ) -> None:
        """Constructor

        Args:
            X (np.array): Design matrix
            y (np.array): Values to predict
            derivative (_type_): Derivative of cost function
            lam (float, optional): Lambda parameter in Ridge regression. Defaults to None.
            reg_param(float, optional) : Regularisation consant for l2-regularisation. Defaults to 0.
        """
        self.X = X
        self.y = y
        self.derivative = derivative
        self.parameters = X.shape[1]
        self.n = X.shape[0]

        self.lam = lam
        self.beta = np.random.randn(self.parameters, 1)
        self.iterations = 0
        self.reg_param = reg_param

        self.data_indices = np.arange(self.n)

    def learning_rate(self, t: float) -> float:
        """Computes decreasing learning rate for mini-batch method.

        Args:
            t (float): Denotes current batch.

        Returns:
            float: learning rate.
        """
        t0 = 1.0
        t1 = 10
        return t0 / (t + t1)

    def gradient_descent_plain(
        self, max_iter: int, learning_rate: float, tol: float = 1e-8
    ) -> None:
        """Performs plain SGD using a fixed number of training iterations, of until distance between betas
           in iterations are smaller than tol.

        Args:
            max_iter (int): Maximum number of iterations.
            learning_rate (float): Learning rate.
            tol (float, optional): Tolerance between current and last beta. Defaults to 1e-8.
        """
        self.iterations = 0
        self.beta = np.random.randn(self.parameters, 1)
        beta_old = self.beta + 10

        while self.iterations < max_iter and np.linalg.norm(self.beta - beta_old) > tol:
            gradient = self.derivative(self.y, self.X, self.beta, self.lam)

            gradient += self.reg_param * self.beta
            beta_old = self.beta.copy()

            self.beta -= learning_rate * gradient

            self.iterations += 1

    def gradient_descent_momentum(
        self, max_iter: int, learning_rate: float, momentum: float, tol: float = 1e-8
    ) -> None:
        """Performs SGD with momentum using a fixed number of training iterations, of until distance between betas
           in iterations are smaller than tol.

        Args:
            max_iter (int): Maximum number of iterations.
            learning_rate (float): Learning rate.
            momentum (float): Momentum coefficient.
            tol (float, optional): Tolerance between current and last beta. Defaults to 1e-8.
        """
        self.iterations = 0
        change = 0.0
        self.beta = np.random.randn(self.parameters, 1)
        beta_old = self.beta + 10

        while self.iterations < max_iter and np.linalg.norm(self.beta - beta_old) > tol:
            gradient = self.derivative(self.y, self.X, self.beta, self.lam)

            new_change = learning_rate * gradient + momentum * change
            beta_old = self.beta.copy()

            self.beta = self.beta - new_change

            change = new_change.copy()
            self.iterations += 1

    def gradient_descent_mini_batch(
        self,
        learning_rate: float,
        epochs: int,
        batches: int,
    ) -> None:
        """Perform SGD using mini-batches with decreasing learning rate.

        Args:
            learning_rate (float): Learning rate.
            epochs (int): Number of epochs.
            batches (int): Batches per epoch.
            with_momentum (bool): Decides if momentum is added to SGD.
            momentum (float): Momentum coefficient.
        """
        m = int(self.n / batches)

        self.beta = np.random.randn(self.parameters, 1)

        for epoch in range(1, epochs + 1):
            for i in range(m):
                idx = np.random.choice(self.data_indices, size=batches, replace=False)
                cur_X = self.X[idx]
                cur_y = self.y[idx]

                t = epoch * m + i
                gradient = self.derivative(cur_y, cur_X, self.beta, self.lam)
                gradient += self.reg_param * self.beta

                self.beta -= self.learning_rate(t) * gradient

    def gradient_descent_AdaGrad(
        self,
        learning_rate: float,
        epochs: int,
        batches: int,
        with_momentum: bool = False,
        momentum: float = None,
        tol: float = 1e-8,
    ) -> None:
        """Performs SGD method ADAgrad with or without momentum.

        Args:
            learning_rate (float):          Learning rate.
            epochs (int):                   Number of epochs to compute.
            batches (int):                  Batch size.
            with_momentum (bool, optional): Decisides wheter or not to use momentum. Defaults to False.
            momentum (float, optional):     Momentum constant. Defaults to None.
            tol (float, optional):          Convergence tolerance. Defaults to 1e-8.
        """
        m = int(self.n / batches)
        self.beta = np.random.randn(self.parameters, 1)
        beta_old = self.beta + 10
        self.iterations = 0
        change = 0
        delta = 1e-8
        for j in tqdm(range(epochs), desc="Running AdaGrad"):
            Giter = 0.0
            if np.linalg.norm(self.beta - beta_old) <= tol:
                break

            beta_old = self.beta.copy()
            for i in range(m):
                idx = np.random.choice(self.data_indices, size=batches, replace=False)
                cur_X = self.X[idx]
                cur_y = self.y[idx]

                gradient = self.derivative(cur_y, cur_X, self.beta, self.lam)
                gradient += self.reg_param * self.beta

                Giter += gradient * gradient
                update = gradient * learning_rate / (delta + np.sqrt(Giter))

                if with_momentum:
                    new_change = update + momentum * change
                    self.beta = self.beta - new_change
                    change = new_change

                else:
                    self.beta -= update

            self.iterations += 1

    def gradient_descent_RMSprop(
        self,
        learning_rate: float,
        epochs: int,
        batches: int,
        rho: float,
        tol: float = 1e-8,
    ) -> None:
        """Performs SGD using RMSprop to tune learning rate.

        Args:
            learning_rate (float):  Learning rate.
            epochs (int):           Number of epochs.
            batches (int):          Batches per epoch.
            rho (float):            Rho parameter used in RMSprop.
            tol (float, optional):  Convergence tolerance. Defaults to 1e-8
        """
        m = int(self.n / batches)
        self.beta = np.random.randn(self.parameters, 1)
        beta_old = self.beta + 10
        self.iterations = 0

        delta = 1e-8
        for j in tqdm(range(epochs), desc="Running RMSprop"):
            Giter = 0.0
            if np.linalg.norm(self.beta - beta_old) <= tol:
                break

            beta_old = self.beta.copy()
            for i in range(m):
                idx = np.random.choice(self.data_indices, size=batches, replace=False)
                cur_X = self.X[idx]
                cur_y = self.y[idx]

                gradient = self.derivative(cur_y, cur_X, self.beta, self.lam)
                gradient += self.reg_param * self.beta

                Giter = rho * Giter + (1 - rho) * gradient * gradient
                update = gradient * learning_rate / (delta + np.sqrt(Giter))
                self.beta -= update

            self.iterations += 1

    def gradient_descent_ADAM(
        self,
        learning_rate: float,
        epochs: int,
        batches: int,
        beta1: float,
        beta2: float,
        tol: float = 1e-8,
    ) -> None:
        """Performs SGD using ADAM to tune learning parameter.

        Args:
            learning_rate (float):  Learning rate.
            epochs (int):           Number of epochs.
            batches (int):          Batches per epoch.
            beta1 (float):          Beta1 parameter in ADAM.
            beta2 (float):          Beta2 parameter in ADAM.
            tol (float, optional):  Convergence tolerance. Defaults to 1e-8
        """
        m = int(self.n / batches)

        delta = 1e-8
        self.beta = np.random.randn(self.parameters, 1)
        beta_old = self.beta + 10
        self.iterations = 0

        for j in range(1, epochs + 1):
            first_moment = 0.0
            second_moment = 0.0

            if np.linalg.norm(self.beta - beta_old) <= tol:
                break

            beta_old = self.beta.copy()
            for i in range(m):
                idx = np.random.choice(self.data_indices, size=batches, replace=False)
                # random_index = batches * np.random.randint(m)
                cur_X = self.X[idx]
                cur_y = self.y[idx]
                # cur_X = self.X[random_index : random_index + batches]
                # cur_y = self.y[random_index : random_index + batches]

                gradient = self.derivative(cur_y, cur_X, self.beta, self.lam)
                gradient += self.reg_param * self.beta

                first_moment = beta1 * first_moment + (1 - beta1) * gradient
                second_moment = (
                    beta2 * second_moment + (1 - beta2) * gradient * gradient
                )

                first_term = first_moment / (1.0 - beta1**j)
                second_term = second_moment / (1.0 - beta2**j)

                update = learning_rate * first_term / (np.sqrt(second_term) + delta)

                self.beta -= update

            self.iterations += 1

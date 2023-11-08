import autograd.numpy as np
from SGD_methods import SGD
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


def sigmoid(x: np.array) -> np.array:
    """Computes the sigmoid function."""
    return 1 / (1 + np.exp(-x))


def accuracy(t: np.array, y: np.array) -> float:
    """Computes the accuracy score of t an y.

    Args:
        t (np.array): Truth values.
        y (np.array): Predicted values

    Returns:
        float: Accuracy score.
    """

    n = t.size
    print(t.shape, y.shape)
    return np.sum(t.astype("int") == y.astype("int")) / n


def cross_entropy_derivative(
    y: np.array, X: np.array, beta: np.array, lam: float = None
) -> np.array:
    """Computes the derivate of the logistic regression cost function.

    Args:
        y (np.array): Array of y-values.
        X (np.array): Design matrix.
        beta (np.array): Array of beta coefficients.
        lam (float): Dummy variable to make code work with Ridge regression. Defaults to None.

    Returns:
        np.array: derivate of the logistic regression cost function.
    """
    n = X.shape[0]
    return -X.T @ (y - sigmoid(X @ beta)) / n


class LogReg(SGD):
    """Logistic regression class with all SGD methods."""

    def predict(self, X: np.array, treshold: float = 0.5) -> np.array:
        """Predicts if each row in X is in class 0 or 1.

        Args:
            X (np.array): Design matrix
            treshold (float, optional):
            Treshold for 0 and 1 class. If probability is above treshold 1 is predicted,
             otherwise 0 is predicted. Defaults to 0.5.

        Returns:
            np.array: Array of 1 and 0, showin class for each row in X.
        """
        return np.where(X @ self.beta >= treshold, 1, 0)


if __name__ == "__main__":
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
    X = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets

    y = np.nan_to_num(y.to_numpy())
    X = np.nan_to_num(X.to_numpy())

    y = (y - 2) / 2

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    model = LogReg(X_train, Y_train, cross_entropy_derivative)
    model.gradient_descent_ADAM(0.01, 10000, 40, 0.9, 0.999)

    results = model.predict(X_test)
    print(accuracy(results, Y_test))

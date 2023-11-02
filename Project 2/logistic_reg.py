import autograd.numpy as np
from SGD_methods import SGD
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy(t, y):
    n = t.size
    print(t.shape, y.shape)
    return np.sum(t.astype("int") == y.astype("int")) / n


def cross_entropy_derivative(y: np.array, X: np.array, beta: np.array, lam: float):
    n = X.shape[0]
    # print((X @ beta).shape)
    return -X.T @ (y - sigmoid(X @ beta)) / n


class LogReg(SGD):
    def predict(self, X, treshold=0.5):
        return np.where(X @ self.beta >= treshold, 1, 0)


if __name__ == "__main__":
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
    X = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets

    y = np.nan_to_num(y.to_numpy())
    X = np.nan_to_num(X.to_numpy())

    y = (y - 2) / 2
    # y = np.array([(1 - i, i) for i in y]).reshape(X.shape[0], 2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    model = LogReg(X_train, Y_train, cross_entropy_derivative)
    model.gradient_descent_ADAM(0.01, 10000, 40, 0.9, 0.999)

    results = model.predict(X_test)
    # print(results, Y_train)
    print(accuracy(results, Y_test))

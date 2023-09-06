import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


class GeneralRegression:
    """Master class for all regression problems"""

    def __init__(
        self, x: np.array, y: np.array, z: np.array, degree: int, treshold: float
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        # self.n = y.size
        self.treshold = treshold

        self.predicted_train = None
        self.predicted_test = None

        self.degree = degree

        self.make_design_matrix(x, y, degree)
        self.split_training_and_test(treshold)

    def make_design_matrix(self, x: np.array, y: np.array, degree: int) -> None:
        self.nr_of_params = (degree + 1) * (degree + 1) - 1
        self.X = np.zeros((x.size, (degree + 1) * (degree + 1)))
        idx = 0

        # If degree = 1, we want [x, y]
        # If degree = 2, we want [x,y, x^2, y^2, xy]

        for j in range(0, degree + 1):
            for i in range(0, degree + 1):
                self.X[:, idx] = (x ** (i)) * (y ** (j))
                idx += 1

        self.X = self.X[:, 1:]
        # self.X = preprocessing.MinMaxScaler().fit_transform(self.X)

    def split_training_and_test(self, treshold: float):
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(
            self.X, self.z, test_size=treshold
        )

    def MSE(self, on_training: bool) -> float:
        if on_training:
            return (
                np.sum((self.z_train - self.predicted_train) ** 2) / self.z_train.size
            )

    def R2(self):
        pass


class OSLpredictor(GeneralRegression):
    def __init__(
        self, x: np.array, y: np.array, f: np.array, degree: int, treshold: float
    ) -> None:
        super().__init__(x, y, f, degree, treshold)

    def compute_parameters(self):
        self.params = (
            np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T
        ) @ self.z_train


class Ridgepredictor(GeneralRegression):
    def __init__(
        self, x: np.array, y: np.array, f: np.array, degree: int, treshold: float
    ) -> None:
        super().__init__(x, y, f, degree, treshold)

    def compute_parameters(self, lam: float):
        n = self.X_train.shape[1]
        # Formula from lecture
        self.params = (
            np.linalg.inv(self.X_train.T @ self.X_train + lam * np.eye(n))
            @ self.X_train.T
        ) @ self.z_train


if __name__ == "__main__":
    n = 100

    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    # print((x * y**0).flatten())
    test = OSLpredictor(x.flatten(), y.flatten(), z.flatten(), 6, 0.2)
    test.compute_parameters()
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        x,
        y,
        (test.X @ test.params).reshape((20, 20)),
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()
    # print(
    #    np.sum(((test.X_train @ test.params) - test.z_train) ** 2) / test.z_train.size
    # )

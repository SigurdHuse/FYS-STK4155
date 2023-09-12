import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
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
        self,
        x: np.array,
        y: np.array,
        z: np.array,
        degree: int,
        treshold: float,
        scale: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        # self.n = y.size
        self.treshold = treshold
        self.scale = scale

        self.predicted_train = None
        self.predicted_test = None

        self.degree = degree

        self.make_design_matrix(x, y, degree)
        self.split_training_and_test(treshold)
        if scale:
            self.scale_data()

    def make_design_matrix(self, x: np.array, y: np.array, degree: int) -> None:
        self.nr_of_params = (degree + 1) * (degree + 2) // 2
        self.X = np.zeros((x.size, self.nr_of_params))
        idx = 0

        # If degree = 1, we want [x, y]
        # If degree = 2, we want [x,y, x^2, y^2, xy]

        for j in range(0, degree + 1):
            for i in range(0, degree + 1):
                if i + j > degree:
                    break
                self.X[:, idx] = (x ** (i)) * (y ** (j))
                # self.X[:, idx] -= np.mean(self.X[:, idx])
                idx += 1

        # We do not compute beta_0
        self.X = self.X[:, 1:]

    def split_training_and_test(self, treshold: float):
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

    def scale_data(self):
        self.fitter = preprocessing.StandardScaler()

        self.fitter.fit(self.X_train)
        self.fitter.transform(self.X_train)
        self.fitter.transform(self.X)
        self.fitter.transform(self.X_test)


class OSLpredictor(GeneralRegression):
    def compute_parameters(self):
        self.params = (
            np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T
        ) @ self.z_train


class Ridgepredictor(GeneralRegression):
    def compute_parameters(self, lam: float):
        n = self.X_train.shape[1]
        # Formula from lecture
        self.params = (
            np.linalg.inv(self.X_train.T @ self.X_train + lam * np.eye(n))
            @ self.X_train.T
        ) @ self.z_train


class Lassopredictor(GeneralRegression):
    def compute_parameters(self, alpha: float) -> None:
        clf = linear_model.Lasso(alpha=alpha)
        clf.fit(self.X_train, self.z_train)
        self.params = clf.coef_


if __name__ == "__main__":
    n = 100

    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    # print((x * y**0).flatten())
    test = OSLpredictor(x.flatten(), y.flatten(), z.flatten(), 5, 0.2)
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

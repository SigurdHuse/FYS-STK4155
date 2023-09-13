import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
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
        else:
            return np.sum((self.z_test - self.predicted_test) ** 2) / self.z_test.size

    def R2(self, on_training: bool) -> float:
        if on_training:
            return 1 - np.sum((self.z_train - self.predicted_train) ** 2) / np.sum(
                ((self.z_train - np.mean(self.z_train)) ** 2)
            )
        else:
            return 1 - np.sum((self.z_test - self.predicted_test) ** 2) / np.sum(
                ((self.z_test - np.mean(self.z_test)) ** 2)
            )

    def scale_data(self):
        self.fitter = preprocessing.StandardScaler()

        self.fitter.fit(self.X_train)
        self.fitter.transform(self.X_train)
        self.fitter.transform(self.X_test)

    def predict_test(self):
        self.predicted_test = self.X_test @ self.params

    def predict_train(self):
        self.predicted_train = self.X_train @ self.params

    def predict_entire_dataset(self) -> np.array:
        if self.scale:
            self.fitter.transform(self.X)
        return self.X @ self.params

    def bootstrap(self, nr_of_its: int, lam: float) -> None:
        nr_of_its = int(nr_of_its)
        self.bootstrap_results = np.zeros((self.X_test.shape[0], nr_of_its))

        for i in range(nr_of_its):
            X_, Y_ = resample(self.X_train, self.z_train)

            self.compute_parameters(lam)
            self.bootstrap_results[:, i] = self.predict_test()

    def cross_validation(self, nr_of_groups: int, lam: float) -> None:
        n = self.X.shape[0]
        self.indexes = np.random.permutation(n)
        self.results = np.zeros(nr_of_groups)

        self.part = [0] * (nr_of_groups + 1)

        inc = n // nr_of_groups
        extra = n % nr_of_groups

        for i in range(1, nr_of_groups + 1):
            self.part[i] = self.part[i - 1] + inc + (extra > 0)
            extra -= 1

        for i in range(1, nr_of_groups + 1):
            # print(self.indexes[0 : self.part[i - 1]])
            self.X_train = np.concatenate(
                (
                    self.X[self.indexes[0 : self.part[i - 1]], :],
                    self.X[self.indexes[self.part[i] :], :],
                )
            )

            self.z_train = np.concatenate(
                (
                    self.z[self.indexes[0 : self.part[i - 1]]],
                    self.z[self.indexes[self.part[i] :]],
                )
            )

            self.X_test = self.X[self.indexes[self.part[i - 1] : self.part[i]], :]
            self.z_test = self.z[self.indexes[self.part[i - 1] : self.part[i]]]

            self.scale_data()
            self.compute_parameters(lam)

            self.predict_test()
            self.results[i - 1] = self.MSE(False)

    def compute_parameters(self, lam: float):
        raise NotImplementedError(
            "This method is not supposed to be called from this class"
        )


class OSLpredictor(GeneralRegression):
    def compute_parameters(self, lam=None):
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

    tmp = np.zeros(7)
    for _ in range(1000):
        test.cross_validation(7, 0)
        tmp += test.results

    print(tmp / 1000)
    """ test.compute_parameters()
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
    plt.show() """
    # print(
    #    np.sum(((test.X_train @ test.params) - test.z_train) ** 2) / test.z_train.size
    # )

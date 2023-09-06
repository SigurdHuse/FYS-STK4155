import numpy as np
import matplotlib.pyplot as plt


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


class GeneralRegression:
    """Master class for all regression problems"""

    def __init__(
        self, x: np.array, y: np.array, f: np.array, degree: int, treshold: float
    ) -> None:
        self.x = x
        self.y = y
        self.f = f
        self.treshold = treshold

        self.degree = degree

        self.make_design_matrix(x, y, degree)
        self.split_training_and_test(treshold)

    def make_design_matrix(self, x: np.array, y: np.array, degree: int) -> None:
        self.nr_of_params = (degree + 1) * (degree + 1)
        self.X = np.zeros((x.size, (degree + 1) * (degree + 1)))
        idx = 0

        for i in range(degree + 1):
            for j in range(degree + 1):
                self.X[:, idx] = x ** (i) * y ** (j)
                idx += 1

    def split_training_and_test(self, treshold: float):
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=treshold
        )

    def MSE(self):
        pass

    def R2(self):
        pass


if __name__ == "__main__":
    n = 100
    test = GeneralRegression(np.zeros(n), np.zeros(n), np.zeros(n), 1, 0.2)
    print(test.X_test.shape)

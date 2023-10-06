import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from matplotlib import cm
from tqdm import tqdm


def FrankeFunction(x: np.array, y: np.array) -> np.array:
    """Computes the Franke function for given x- and y-coordinates

    Args:
        x (np.array): Mesh grid containing x-coordinates
        y (np.array): Mesh grid containing y-coordinates

    Returns:
        np.array: 2D-array containing computed values.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


class GeneralRegression:
    """Master class for all regression problems,
    this class is not intended to be used but be inherited from"""

    def __init__(
        self,
        x: np.array,
        y: np.array,
        z: np.array,
        degree: int,
        treshold: float,
        scale: bool = True,
        variance_of_noise: float = -1,
    ) -> None:
        """Constructor for the general regression class

        Args:
            x (np.array): Coordinates in x-direction
            y (np.array): Coordinates in y-direction
            z (np.array): Value model is trying to predict, in z-direction
            degree (int): Degree of polynomials in design matrix
            treshold (float): Treshold to split training and test data
            scale (bool, optional): Bool to decide if data should be scaled. Defaults to True.
        """
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

        if variance_of_noise > 0:
            self.z += np.random.normal(0, variance_of_noise, z.shape)

        if scale:
            self.scale_data()

    def make_design_matrix(self, x: np.array, y: np.array, degree: int) -> None:
        """Makes design matrix for x- and y-coordinates, by multiplying them togheter and making
        polynomials with sum of degrees smaller or equal given degree.

        Args:
            x (np.array): x-coordinates
            y (np.array): y-coordinates
            degree (int): Max degree of constructed polynomials
        """
        self.nr_of_params = (degree + 1) * (degree + 2) // 2
        self.X = np.zeros((x.size, self.nr_of_params), dtype=np.float64)
        idx = 0

        # If degree = 1, we want [x, y]
        # If degree = 2, we want [x,y, x^2, y^2, xy]

        for j in range(0, degree + 1):
            for i in range(0, degree + 1):
                if i + j > degree:
                    break
                self.X[:, idx] = np.power(x, i) * np.power(y, j)
                idx += 1

        # We do not compute beta_0
        self.X = self.X[:, 1:]

    def split_training_and_test(self, treshold: float) -> None:
        """Split input data into training and test data.

        Args:
            treshold (float): Treshold used to split data. For example treshold = 0.2,
                              splits the data into 20% test and 80% training.
        """
        self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(
            self.X, self.z, test_size=treshold
        )

    def MSE(self, on_training: bool) -> float:
        """Computes MSE for training or test data.

        Args:
            on_training (bool): If true MSE is computed on training data, if false MSE is
                                computed on test data.

        Returns:
            float: Computed MSE.
        """
        if on_training:
            return np.mean(np.square(self.z_train - self.predicted_train))
        else:
            return np.mean(np.square(self.z_test - self.predicted_test))

    def R2(self, on_training: bool) -> float:
        """Computes R2-score on training or test data.

        Args:
            on_training (bool): If true MSE is computed on training data, if false MSE is
                                computed on test data.

        Returns:
            float: Computed R2-score.
        """
        if on_training:
            return 1 - np.sum((self.z_train - self.predicted_train) ** 2) / np.sum(
                ((self.z_train - np.mean(self.z_train)) ** 2)
            )
        else:
            return 1 - np.sum((self.z_test - self.predicted_test) ** 2) / np.sum(
                ((self.z_test - np.mean(self.z_test)) ** 2)
            )

    def scale_data(self) -> None:
        """Scales data according to sklearn's StandardScaler using training data."""
        self.fitter = preprocessing.StandardScaler()

        self.fitter.fit(self.X_train)
        self.X_train = self.fitter.transform(self.X_train)
        self.X_test = self.fitter.transform(self.X_test)

    def predict_test(self) -> None:
        """Predicts z-values for test data, using computed parameters of model."""
        self.predicted_test = self.X_test @ self.params + self.z_train.mean()

    def predict_train(self) -> None:
        """Predicts z-values for training data, using computed parameters of model."""
        self.predicted_train = self.X_train @ self.params + self.z_train.mean()

    def predict_entire_dataset(self) -> np.array:
        """Predicts z-values for entire dataset, using computed parameters of model.

        Returns:
            np.array: Predicted z-values
        """
        if self.scale:
            self.X = self.fitter.transform(self.X)
        return self.X @ self.params + self.z_train.mean()

    def bootstrap(self, nr_of_its: int, lam: float) -> None:
        """Performs the resampling technique bootstrap on the training data.

        Args:
            nr_of_its (int): Number of boostrap iterations performed.
            lam (float): Lambda parameter to use in regression method.
        """
        nr_of_its = int(nr_of_its)

        # Predicted z-values for test data is stored in each column
        self.bootstrap_results = np.zeros((self.X_test.shape[0], nr_of_its))

        for i in tqdm(range(nr_of_its), desc="Running bootstrap"):
            X_, z_ = resample(self.X_train, self.z_train)

            self.computer_parameters_from_input(X_, z_, lam)
            self.predict_test()
            self.predicted_test -= self.z_train.mean()
            self.predicted_test += z_.mean()
            # print(self.params)
            self.bootstrap_results[:, i] = self.predicted_test

    def cross_validation(self, nr_of_groups: int, lam: float) -> None:
        n = self.X.shape[0]
        self.indexes = np.random.permutation(n)
        self.results_cross_val = np.zeros(nr_of_groups)

        self.part = [0] * (nr_of_groups + 1)

        inc = n // nr_of_groups
        extra = n % nr_of_groups

        for i in range(1, nr_of_groups + 1):
            self.part[i] = self.part[i - 1] + inc + (extra > 0)
            extra -= 1

        for i in tqdm(range(1, nr_of_groups + 1), desc="Running Cross-validation"):
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
            self.results_cross_val[i - 1] = self.MSE(on_training=False)

    def compute_parameters(self, lam: float) -> None:
        """This method is not supposed to be called from this class, but insted be implemented in
        classes which inherit from the GeneralRegression class.

        Args:
            lam (float): Argument for regression model.

        Raises:
            NotImplementedError: This method is not supposed to be called from this class
        """
        raise NotImplementedError(
            "This method is not supposed to be called from this class"
        )

    def computer_parameters_from_input(
        self, X: np.array, z: np.array, lam=None
    ) -> None:
        raise NotImplementedError(
            "This method is not supposed to be called from this class"
        )


class OLSpredictor(GeneralRegression):
    def compute_parameters(self, lam=None):
        """Computes parameters of model using standard least squared regression.

        Args:
            lam (any, optional): This is only here as GeneralRegression class requires it. Defaults to None.
        """
        self.params = (
            np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T
        ) @ self.z_train

    def computer_parameters_from_input(
        self, X: np.array, z: np.array, lam=None
    ) -> None:
        """Computes parameters of model using standard least squared regression, with given design matrix and prediction targets.

        Args:
            X (np.array): Design matrix
            z (np.array): Values to predict
            lam (any, optional): This is only here as GeneralRegression class requires it. Defaults to None.
        """
        self.params = (np.linalg.pinv(X.T @ X) @ X.T) @ z


class Ridgepredictor(GeneralRegression):
    def compute_parameters(self, lam: float):
        """Computes parameters of model using Ridge regression.

        Args:
            lam (float): Lambda parameter in Ridge regression.
        """
        n = self.X_train.shape[1]
        # Formula from lecture
        self.params = (
            np.linalg.pinv(self.X_train.T @ self.X_train + lam * np.eye(n))
            @ self.X_train.T
        ) @ self.z_train


class Lassopredictor(GeneralRegression):
    def compute_parameters(self, alpha: float) -> None:
        """Computes parameters of model using sklearn's Lasso regression.

        Args:
            alpha (float): Alpha parameter of Lasso regression.
        """
        clf = linear_model.Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        clf.fit(self.X_train, self.z_train)
        self.params = clf.coef_


if __name__ == "__main__":
    n = 100

    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    # print((x * y**0).flatten())
    test = Lassopredictor(x.flatten(), y.flatten(), z.flatten(), 15, 0.2, True)

    """ tmp = np.zeros(7)
    for _ in range(1000):
        test.cross_validation(7, 0)
        tmp += test.results

    print(tmp / 1000) """
    test.compute_parameters(1e-4)
    print(np.var(test.params))
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        x,
        y,
        test.predict_entire_dataset().reshape((20, 20)),
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show()
    # print(
    #    np.sum(((test.X_train @ test.params) - test.z_train) ** 2) / test.z_train.size
    # )

from SGD_methods import SGD
import os
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import SymLogNorm
from sklearn.model_selection import train_test_split
from logistic_reg import LogReg, cross_entropy_derivative
from sklearn.metrics import mean_squared_error, accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn import preprocessing
from matplotlib.colors import SymLogNorm

mpl.rcParams["figure.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 13
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

dir_name = "plots_SGD"
data_dir_name = "data_synthetic"
file_extension = ".pdf"
seed = 2023


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


def derivative_OLS(y: np.array, X: np.array, beta: np.array, lam=None):
    """Computes the derivate with respect to beta of the OLS cost function."""
    # n = X.shape[0]
    # print(X.shape, beta.shape, y.shape)
    # print(np.matmul(X, beta).shape)
    # print((y - X @ beta).shape)
    return -2.0 / y.size * X.T @ (y - X @ beta)


def CostOLS(y: np.array, X: np.array, beta: np.array, lam=None) -> np.array:
    return np.mean((y - X @ beta) ** 2)


def CostRidge(y: np.array, X: np.array, beta: np.array, lam: float) -> np.array:
    n = X.shape[0]
    return np.sum((y - X @ beta) ** 2) / n + lam * beta.T @ beta


def f(x: np.array) -> np.array:
    return 4 + 3 * x + 4 * x**2


def plot_OLS(
    X: np.array,
    y: np.array,
    learning_rates: np.array,
    name: str,
    beta_optimal: np.array,
    method: str,
    X_test: np.array,
    z_test: np.array,
    rho: float = None,
    beta1: float = None,
    beta2: float = None,
    momentum: float = None,
    max_iter: int = int(1e3),
) -> None:
    fig = plt.figure(dpi=250)

    tmp = SGD(X, y, grad(CostOLS, 2))
    results, iterations = np.zeros((2, learning_rates.size))
    optimal_error = mean_squared_error(z_test, X_test @ beta_optimal + y.mean())
    print(f"Optimal error using OLS: {optimal_error : e}")
    for idx, learn in enumerate(learning_rates):
        if method == "plain_GD":
            tmp.gradient_descent_plain(max_iter, learn)
        elif method == "momentum_GD":
            tmp.gradient_descent_momentum(max_iter, learn, momentum)
        elif method == "RMSprop":
            tmp.gradient_descent_RMSprop(learn, int(1e3), 64, rho)
        elif method == "ADAM":
            tmp.gradient_descent_ADAM(learn, int(1e3), 64, beta1, beta2)
        elif method == "AdaGrad":
            if momentum != None:
                tmp.gradient_descent_AdaGrad(learn, int(1e3), 64, True, momentum)
            else:
                tmp.gradient_descent_AdaGrad(learn, int(1e3), 64, False)
        elif method == "MiniBatch":
            tmp.gradient_descent_mini_batch(learn, int(1e3), 64)
        else:
            raise NotImplementedError

        # print(tmp.beta)
        results[idx] = mean_squared_error(z_test, X_test @ tmp.beta + y.mean())
        iterations[idx] = tmp.iterations

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(learning_rates, results, label="Error")
    ax1.plot(
        learning_rates,
        np.full(learning_rates.size, optimal_error),
        label="Optimal Error",
    )

    ax1.grid()
    ax1.set(ylabel="Error", xlabel="Learning rate")
    ax1.legend()

    ax2.plot(learning_rates, iterations, label="Iterations")
    ax2.grid()
    ax2.set(ylabel="Training iterations", xlabel="Learning rate")
    ax2.legend()

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")

    fig.tight_layout()
    plt.savefig(dir_name + "/" + name + file_extension)
    plt.close()


def plot_Ridge(
    X: np.array,
    y: np.array,
    learning_rates: np.array,
    lambdas: np.array,
    name: str,
    method: str,
    X_test: np.array,
    z_test: np.array,
    rho: float = None,
    beta1: float = None,
    beta2: float = None,
    momentum: float = None,
    max_iter: int = int(1e3),
):
    plt.rcParams["figure.figsize"] = [10, 10]
    tmp = SGD(X, y, grad(CostRidge, 2))
    results, iterations = np.zeros((2, lambdas.size, learning_rates.size))

    for i, lam in enumerate(lambdas):
        tmp.lam = lam
        for j, learn in enumerate(learning_rates):
            if method == "plain_GD":
                tmp.gradient_descent_plain(max_iter, learn)
            elif method == "momentum_GD":
                tmp.gradient_descent_momentum(max_iter, learn, momentum)
            elif method == "RMSprop":
                tmp.gradient_descent_RMSprop(learn, int(1e3), 64, rho)
            elif method == "ADAM":
                tmp.gradient_descent_ADAM(learn, int(1e3), 64, beta1, beta2)
            elif method == "AdaGrad":
                if momentum != None:
                    tmp.gradient_descent_AdaGrad(learn, int(1e3), 64, True, momentum)
                else:
                    tmp.gradient_descent_AdaGrad(learn, int(1e3), 64, False)
            else:
                raise NotImplementedError

            results[i, j] = mean_squared_error(z_test, X_test @ tmp.beta + y.mean())
            iterations[i, j] = tmp.iterations

    plt.matshow(results, cmap=cm.coolwarm, norm=SymLogNorm(1e-8))
    for (i, j), z in np.ndenumerate(results):
        plt.text(j, i, "{:0.1e}".format(z), ha="center", va="center")

    plt.colorbar(label="MSE on test data")

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel("Learning rates")
    plt.ylabel(r"$\lambda$")

    plt.xticks(
        ticks=list(range(learning_rates.size)),
        labels=[f"{i : .1e}" for i in learning_rates],
    )
    plt.yticks(
        ticks=list(range(lambdas.size)),
        labels=[f"{i : .1e}" for i in lambdas],
    )

    plt.tick_params(axis="x", rotation=90)

    plt.savefig(dir_name + "/" + name + ".pdf")
    plt.close()


def plot_logistic(
    X: np.array,
    y: np.array,
    learning_rates: np.array,
    reg_param: np.array,
    name: str,
    method: str,
    X_test: np.array,
    z_test: np.array,
    rho: float = None,
    beta1: float = None,
    beta2: float = None,
    momentum: float = None,
    max_iter: int = int(5e3),
):
    plt.rcParams["figure.figsize"] = [10, 10]
    tmp = LogReg(X, y, cross_entropy_derivative)
    results, iterations = np.zeros((2, reg_param.size, learning_rates.size))

    for i, lam in enumerate(reg_param):
        tmp.reg_param = lam
        for j, learn in enumerate(learning_rates):
            if method == "plain_GD":
                tmp.gradient_descent_plain(max_iter, learn)
            elif method == "momentum_GD":
                tmp.gradient_descent_momentum(max_iter, learn, momentum)
            elif method == "RMSprop":
                tmp.gradient_descent_RMSprop(learn, int(5e3), 20, rho)
            elif method == "ADAM":
                tmp.gradient_descent_ADAM(learn, int(1e3), 10, beta1, beta2)
            elif method == "AdaGrad":
                tmp.gradient_descent_AdaGrad(learn, 100, 10)
            else:
                raise NotImplementedError

            results[i, j] = accuracy_score(z_test, tmp.predict(X_test))
            # iterations[i, j] = tmp.iterations

    plt.matshow(results, cmap=cm.coolwarm)
    plt.colorbar(label="Accuracy on test data")

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    plt.xlabel("Learning rates")
    plt.ylabel("Regularisation paramater")

    plt.xticks(
        ticks=list(range(learning_rates.size)),
        labels=[f"{i : .1e}" for i in learning_rates],
    )
    plt.yticks(
        ticks=list(range(reg_param.size)),
        labels=[f"{i : .1e}" for i in reg_param],
    )

    plt.tick_params(axis="x", rotation=90)
    for (i, j), z in np.ndenumerate(results):
        plt.text(j, i, "{:0.3f}".format(z), ha="center", va="center")

    plt.savefig(dir_name + "/" + name + ".pdf")
    plt.close()


def plot_entire_dataset(
    x: np.array, y: np.array, z: np.array, dir_name: str, file_name: str
):
    fig = plt.figure(dpi=250)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")

    ax1.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )

    ax2 = fig.add_subplot(1, 2, 2)
    surf = ax2.matshow(z)
    fig.colorbar(surf, label="z-value")
    fig.tight_layout(pad=3.5)
    # plt.show()
    # fig.legend()
    plt.savefig(dir_name + "/" + file_name)
    plt.close()


if __name__ == "__main__":
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    np.random.seed(seed)

    variance = 0.01
    n = 30

    degree = 5
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    z += np.random.normal(0, variance, z.shape)

    """ plot_entire_dataset(
        x=x, y=y, z=z, dir_name=dir_name, file_name="Franke_function.pdf"
    ) """
    nr_of_params = (degree + 1) * (degree + 2) // 2
    X = np.zeros((x.size, nr_of_params), dtype=np.float64)
    idx = 0

    for j in range(0, degree + 1):
        for i in range(0, degree + 1):
            if i + j > degree:
                break
            X[:, idx] = np.power(x.flatten(), i) * np.power(y.flatten(), j)
            idx += 1

    X = X[:, 1:]
    input = (z.flatten()).reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, input, test_size=0.2)

    scaler_franke = preprocessing.StandardScaler()
    scaler_franke.fit(X_train)

    X_train = scaler_franke.transform(X_train)
    X_test = scaler_franke.transform(X_test)

    beta_optimal_OLS = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train

    # print(np.linalg.norm(z.flatten() - X @ beta_optimal_OLS))
    # Plain OLS
    """ plot_OLS(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 40),
        name="plain_SGD_OLS",
        beta_optimal=beta_optimal_OLS,
        method="plain_GD",
        X_test=X_test,
        z_test=Y_test,
    )

    # Momentum OLS
    plot_OLS(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 40),
        name="momentum_SGD_OLS",
        beta_optimal=beta_optimal_OLS,
        method="momentum_GD",
        momentum=0.2,
        X_test=X_test,
        z_test=Y_test,
    )

    # AdaGrad momnetum OLS
    plot_OLS(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 40),
        name="AdaGrad_SGD_OLS_momentum",
        beta_optimal=beta_optimal_OLS,
        method="AdaGrad",
        momentum=0.1,
        X_test=X_test,
        z_test=Y_test,
    )

    # AdaGrad no momnetum OLS
    plot_OLS(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 40),
        name="AdaGrad_SGD_OLS_no_momentum",
        beta_optimal=beta_optimal_OLS,
        method="AdaGrad",
        X_test=X_test,
        z_test=Y_test,
    )

    # RMSprop OLS
    plot_OLS(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 40),
        name="RMSprop_SGD_OLS",
        beta_optimal=beta_optimal_OLS,
        method="RMSprop",
        rho=0.99,
        X_test=X_test,
        z_test=Y_test,
    )

    # ADAM OLS
    plot_OLS(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 0.1, 40),
        name="ADAM_SGD_OLS",
        beta_optimal=beta_optimal_OLS,
        method="ADAM",
        beta1=0.9,
        beta2=0.999,
        X_test=X_test,
        z_test=Y_test,
    ) """

    # Plain SGD Ridge
    """ plot_Ridge(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 10),
        lambdas=np.logspace(-6, -1, 10),
        name="plain_SGD_Ridge",
        method="plain_GD",
        X_test=X_test,
        z_test=Y_test,
    ) """

    # Momentum SGD Rige
    """ plot_Ridge(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 10),
        lambdas=np.logspace(-6, -1, 10),
        name="momentum_SGD_Ridge",
        method="momentum_GD",
        momentum=0.1,
        X_test=X_test,
        z_test=Y_test,
    ) """

    """ plot_Ridge(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 10),
        lambdas=np.logspace(-6, -1, 10),
        name="RMSprop_SGD_Ridge",
        method="RMSprop",
        rho=0.99,
        X_test=X_test,
        z_test=Y_test,
    ) """
    """ plot_Ridge(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 10),
        lambdas=np.logspace(-6, -1, 10),
        name="AdaGrad_no_momnetum_SGD_Ridge",
        method="AdaGrad",
        X_test=X_test,
        z_test=Y_test,
    ) """

    """ plot_Ridge(
        X=X_train,
        y=Y_train,
        learning_rates=np.linspace(1e-5, 1e-2, 10),
        lambdas=np.logspace(-6, -1, 10),
        name="ADAM_SGD_Ridge",
        method="ADAM",
        beta1=0.9,
        beta2=0.999,
        X_test=X_test,
        z_test=Y_test,
    ) """

    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
    X_cancer = breast_cancer_wisconsin_original.data.features
    y_cancer = breast_cancer_wisconsin_original.data.targets

    y_cancer = np.nan_to_num(y_cancer.to_numpy())
    X_cancer = np.nan_to_num(X_cancer.to_numpy())

    y_cancer = (y_cancer - 2) / 2
    # y_cancer = np.array([(1 - i, i) for i in y_cancer]).reshape(X_cancer.shape[0], 2)

    X_train_cancer, X_test_cancer, Y_train_cancer, Y_test_cancer = train_test_split(
        X_cancer, y_cancer, test_size=0.2
    )

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_cancer)
    X_train_cancer = scaler.transform(X_train_cancer)
    X_test_cancer = scaler.transform(X_test_cancer)

    plot_logistic(
        X=X_train_cancer,
        y=Y_train_cancer,
        learning_rates=np.linspace(1e-3, 0.1, 10),
        reg_param=np.linspace(0, 0.5, 10),
        name="Logistic",
        method="plain_GD",
        X_test=X_test_cancer,
        z_test=Y_test_cancer,
    )

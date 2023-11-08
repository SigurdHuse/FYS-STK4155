from FFNN import *
from sklearn.neural_network import MLPRegressor
import matplotlib as mpl
import os
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from sklearn import preprocessing
from matplotlib.colors import SymLogNorm

mpl.rcParams["figure.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 13
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

dir_name = "plots_FFNN"
data_dir_name = "data_synthetic"
file_extension = ".pdf"
seed = 2023


def Relu(x: np.array) -> np.array:
    """Computes the Relu function."""
    return np.maximum(x, 0)


def deriv_Relu(x: np.array) -> np.array:
    """Computes the derivative of the Relu function."""
    return (x > 0) * 1


def Leaky_Relu(x: np.array) -> np.array:
    """Computes the leaky ReLu function"""
    alpha = 1e-2
    return np.maximum(x, 0) + alpha * np.minimum(0, x)


def deriv_Leaky_Relu(x: np.array) -> np.array:
    """Computes the derivative of the leaky ReLu function."""
    alpha = 1e-2
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def tanh(x: np.array) -> np.array:
    """Computes the tanh function."""
    return np.tanh(x)


def deriv_tanh(x: np.array) -> np.array:
    """Computes the derivative of the tanh function."""
    return 1 / (1 + x * x)


def compare_sklearn_and_our_code(
    X: np.array,
    y: np.array,
    X_test: np.array,
    Y_test: np.array,
    learning_rates: np.array,
    layers: int,
    neurons_per_layer: int,
    name: str,
) -> None:
    """Makes a plot comparing sklearn's and our our implementation of FFNNs.

    Args:
        X (np.array):                   Design matrix.
        y (np.array):                   Targets.
        X_test (np.array):              Test design matrix.
        Y_test (np.array):              Test targets.
        learning_rates (np.array):      Array of learning rates to try.
        layers (int):                   Number of hidden layers in FFNN.
        neurons_per_layer (int):        Number of neurons per hidden layer.
        name (str):                     Filename to save plot as.
    """
    res1, res2 = np.zeros((2, learning_rates.size))
    for idx, learn in enumerate(learning_rates):
        our_model = FFNN(
            X_data=X,
            Y_data=y,
            layers=layers,
            activation_function=sigmoid,
            activation_function_derivative=deriv_sigmoid,
            hidden_neurons=neurons_per_layer,
            epochs=1000,
            batch_size=100,
        )
        our_model.learning_rate = learn
        dnn = MLPRegressor(
            hidden_layer_sizes=[neurons_per_layer for i in range(layers)],
            alpha=0.0,
            learning_rate_init=learn,
            activation="logistic",
            max_iter=1000,
            solver="sgd",
            batch_size=100,
        )
        dnn.fit(X, y.ravel())
        our_model.train()

        res1[idx] = mean_squared_error(Y_test, dnn.predict(X_test))

        test_predict = our_model.predict(X_test)
        res2[idx] = mean_squared_error(Y_test, test_predict)

    plt.plot(learning_rates, res1, label="SKlearn")
    plt.plot(learning_rates, res2, label="Our model")
    plt.ylabel("Mean squared error")
    plt.xlabel("Learning rate")
    plt.grid()
    plt.xscale("log")
    plt.legend()
    plt.savefig(dir_name + "/" + name + file_extension)
    plt.close()


def compare_activation_functions(
    X: np.array,
    y: np.array,
    X_test: np.array,
    Y_test: np.array,
    learning_rates: np.array,
    lambdas: np.array,
    layers: int,
    neurons_per_layer: int,
    name: str,
) -> None:
    """Compares MSE achieved using different activation functions for the regression problem.

    Args:
        X (np.array):               Design matrix.
        y (np.array):               Targets.
        X_test (np.array):          Test design matrix.
        Y_test (np.array):          Test targets.
        learning_rates (np.array):  Array of learning rates to try.
        lambdas (np.array):         Array of regularisation parameters to try.
        layers (int):               Number of hidden layers.
        neurons_per_layer (int):    Number of nodes per hidden layers.
        name (str):                 Filename to save plot as.
    """
    plt.rcParams["figure.figsize"] = [10, 10]
    fig, axs = plt.subplots(2, 2)
    fig.set_dpi(250)

    activ = [
        (sigmoid, deriv_sigmoid),
        (Relu, deriv_Relu),
        (tanh, deriv_tanh),
        (Leaky_Relu, deriv_Leaky_Relu),
    ]

    i, j = 0, 0

    results = np.zeros((learning_rates.size, lambdas.size))

    for cur in activ:
        for k, learn in enumerate(learning_rates):
            for l, lam in enumerate(lambdas):
                # print(learn, lam)
                model = FFNN(
                    X_data=X,
                    Y_data=y,
                    layers=layers,
                    activation_function=cur[0],
                    activation_function_derivative=cur[1],
                    learning_rate=learn,
                    hidden_neurons=neurons_per_layer,
                    lam=lam,
                    epochs=1000,
                    batch_size=10,
                )
                model.train()
                results[k, l] = mean_squared_error(Y_test, model.predict(X_test))
        im = axs[i, j].matshow(results, cmap=cm.coolwarm, norm=SymLogNorm(1e-8))

        for (e, r), z in np.ndenumerate(results):
            axs[i, j].text(r, e, "{:.1e}".format(z), ha="center", va="center")

        i += 1
        if i == 2:
            j += 1
            i = 0

    for i in range(2):
        for j in range(2):
            axs[i, j].xaxis.set_major_locator(MultipleLocator(1))
            axs[i, j].yaxis.set_major_locator(MultipleLocator(1))
            axs[i, j].set_xticks(ticks=list(range(learning_rates.size)))
            axs[i, j].set_xticklabels(
                labels=[f"{i : .1e}" for i in learning_rates], rotation=90
            )
            axs[i, j].set_yticks(ticks=list(range(lambdas.size)))
            axs[i, j].set_yticklabels(
                labels=[f"{i : .1e}" for i in lambdas],
            )
            axs[i, j].xaxis.set_ticks_position("bottom")
            axs[i, j].set_ylabel(r"$\lambda$")
            axs[i, j].set_xlabel("Learning rate")
            axs[i, j].yaxis.set_label_position("right")

    axs[0, 0].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[0, 0].set_title("Sigmoid")
    axs[0, 1].set_title("ReLu")
    axs[1, 0].set_title("Tanh")
    axs[1, 1].set_title("Leaky ReLu")

    fig.colorbar(im, ax=axs.ravel().tolist(), label="MSE on test data")
    # fig.tight_layout()
    plt.savefig(dir_name + "/" + name + ".pdf")
    plt.close()


def compare_networks_cancer(
    X: np.array,
    y: np.array,
    X_test: np.array,
    Y_test: np.array,
    learning_rates: np.array,
    lambdas: np.array,
    layers: int,
    neurons_per_layer: np.array,
    name: str,
    activation,
    deriv_activation,
) -> None:
    """Compares different networks solving the classification problem.

    Args:
        X (np.array):                   Design matrix.
        y (np.array):                   Targets.
        X_test (np.array):              Test design matrix.
        Y_test (np.array):              Test targets.
        learning_rates (np.array):      Array of learning rates to try.
        lambdas (np.array):             Array of regularisation parameters to try.
        layers (int):                   Number of hidden layers.
        neurons_per_layer (np.array):   Number of nodes per hidden layer.
        name (str):                     Filename to save plot as.
        activation (function):          Activation function.
        deriv_activation (function):    Derivative of activation function.
    """
    plt.rcParams["figure.figsize"] = [10, 10]
    fig, axs = plt.subplots(2, 2)
    fig.set_dpi(250)

    assert (
        neurons_per_layer.size == 4
    ), "neurons_per_layer needs to contain exactly 4 elements"

    i, j = 0, 0

    results = np.zeros((learning_rates.size, lambdas.size))

    for nodes in neurons_per_layer:
        for k, learn in enumerate(learning_rates):
            for l, lam in enumerate(lambdas):
                # print(learn, lam)
                model = FFNN(
                    X_data=X,
                    Y_data=y,
                    layers=layers,
                    activation_function=activation,
                    activation_function_derivative=deriv_activation,
                    learning_rate=learn,
                    hidden_neurons=nodes,
                    lam=lam,
                    epochs=1000,
                    batch_size=20,
                    n_categories=2,
                    softmax_last_layer=True,
                )
                model.train()
                results[k, l] = accuracy(Y_test, model.predict(X_test))

        im = axs[i, j].matshow(results, cmap=cm.coolwarm)

        for (e, r), z in np.ndenumerate(results):
            axs[i, j].text(r, e, "{:0.3f}".format(z), ha="center", va="center")

        i += 1
        if i == 2:
            j += 1
            i = 0

    for i in range(2):
        for j in range(2):
            axs[i, j].xaxis.set_major_locator(MultipleLocator(1))
            axs[i, j].yaxis.set_major_locator(MultipleLocator(1))
            axs[i, j].set_xticks(ticks=list(range(learning_rates.size)))
            axs[i, j].set_xticklabels(
                labels=[f"{i : .1e}" for i in learning_rates], rotation=90
            )
            axs[i, j].set_yticks(ticks=list(range(lambdas.size)))
            axs[i, j].set_yticklabels(
                labels=[f"{i : .1e}" for i in lambdas],
            )
            axs[i, j].xaxis.set_ticks_position("bottom")
            axs[i, j].set_ylabel(r"$\lambda$")
            axs[i, j].set_xlabel("Learning rate")
            axs[i, j].yaxis.set_label_position("right")

    axs[0, 0].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[0, 0].set_title(f"{neurons_per_layer[0]} Hidden nodes")
    axs[0, 1].set_title(f"{neurons_per_layer[1]} Hidden nodes")
    axs[1, 0].set_title(f"{neurons_per_layer[2]} Hidden nodes")
    axs[1, 1].set_title(f"{neurons_per_layer[3]} Hidden nodes")

    fig.colorbar(im, ax=axs.ravel().tolist(), label="Accuracy on test data")
    # fig.tight_layout()
    plt.savefig(dir_name + "/" + name + ".pdf")
    plt.close()


if __name__ == "__main__":
    np.random.seed(seed)

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    n = 30
    variance = 0.01
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)
    z += np.random.normal(0, variance, z.shape)

    X = np.zeros((x.size, 2), dtype=np.float64)
    X[:, 0] = x.flatten()
    X[:, 1] = y.flatten()

    input = (z.flatten()).reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, input, test_size=0.2)

    scaler_franke = preprocessing.StandardScaler()
    scaler_franke.fit(X_train)

    X_train = scaler_franke.transform(X_train)
    X_test = scaler_franke.transform(X_test)

    compare_sklearn_and_our_code(
        X=X_train,
        y=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        learning_rates=np.linspace(1e-6, 1e-4, 20),
        layers=2,
        neurons_per_layer=50,
        name="comparison_sklearn",
    )

    compare_activation_functions(
        X=X_train,
        y=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        learning_rates=np.linspace(1e-6, 1e-4, 5),
        lambdas=np.linspace(0, 0.4, 5),
        layers=2,
        neurons_per_layer=60,
        name="activation_function_comparison",
    )

    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
    X_cancer = breast_cancer_wisconsin_original.data.features
    y_cancer = breast_cancer_wisconsin_original.data.targets

    y_cancer = np.nan_to_num(y_cancer.to_numpy())
    X_cancer = np.nan_to_num(X_cancer.to_numpy())

    y_cancer = (y_cancer - 2) / 2
    y_cancer = np.array([(1 - i, i) for i in y_cancer]).reshape(X_cancer.shape[0], 2)

    X_train_cancer, X_test_cancer, Y_train_cancer, Y_test_cancer = train_test_split(
        X_cancer, y_cancer, test_size=0.2
    )

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_cancer)
    X_train_cancer = scaler.transform(X_train_cancer)
    X_test_cancer = scaler.transform(X_test_cancer)

    for layers in [2]:
        compare_networks_cancer(
            X=X_train_cancer,
            y=Y_train_cancer,
            X_test=X_test_cancer,
            Y_test=Y_test_cancer,
            learning_rates=np.linspace(1e-6, 1e-4, 5),
            lambdas=np.linspace(0, 0.5, 5),
            layers=layers,
            neurons_per_layer=np.array([10, 50, 80, 120]),
            name=f"Cancer_compare_Relu",
            activation=Relu,
            deriv_activation=deriv_Relu,
        )

        compare_networks_cancer(
            X=X_train_cancer,
            y=Y_train_cancer,
            X_test=X_test_cancer,
            Y_test=Y_test_cancer,
            learning_rates=np.linspace(1e-6, 1e-4, 5),
            lambdas=np.linspace(0, 0.5, 5),
            layers=layers,
            neurons_per_layer=np.array([10, 50, 80, 120]),
            name=f"Cancer_compare_Leaky_Relu",
            activation=Leaky_Relu,
            deriv_activation=deriv_Leaky_Relu,
        )

        compare_networks_cancer(
            X=X_train_cancer,
            y=Y_train_cancer,
            X_test=X_test_cancer,
            Y_test=Y_test_cancer,
            learning_rates=np.linspace(1e-6, 1e-4, 5),
            lambdas=np.linspace(0, 0.5, 5),
            layers=layers,
            neurons_per_layer=np.array([10, 50, 80, 120]),
            name=f"Cancer_compare_sigmoid",
            activation=sigmoid,
            deriv_activation=deriv_sigmoid,
        )
        compare_networks_cancer(
            X=X_train_cancer,
            y=Y_train_cancer,
            X_test=X_test_cancer,
            Y_test=Y_test_cancer,
            learning_rates=np.linspace(1e-6, 1e-4, 5),
            lambdas=np.linspace(0, 0.5, 5),
            layers=layers,
            neurons_per_layer=np.array([10, 50, 80, 120]),
            name=f"Cancer_compare_tanh",
            activation=tanh,
            deriv_activation=deriv_tanh,
        )

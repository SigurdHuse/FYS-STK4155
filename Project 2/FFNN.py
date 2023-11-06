import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
from plotter_SGD import FrankeFunction


def f(x: np.array) -> np.array:
    return 4 + 3 * x + 4 * x**2


def CostOLS(y: np.array, X: np.array) -> np.array:
    n = X.shape[0]
    return np.sum((y - X) ** 2) / 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def accuracy(t, y):
    n = t.size
    return np.sum(t.astype("int") == y.astype("int")) / n


def f(x):
    return 4 + 3 * x + 4 * x**2


class FFNN:
    """Feed-forward neural network."""

    def __init__(
        self,
        X_data: np.array,
        Y_data: np.array,
        layers: int,
        activation_function,
        activation_function_derivative,
        hidden_neurons: int = 50,
        n_categories: int = 1,
        epochs: int = 10,
        batch_size: int = 100,
        learning_rate: float = 0.1,
        lam: float = 0.0,
        softmax_last_layer: bool = False,
    ) -> None:
        """Constructor.

        Args:
            X_data (np.array):                          Input data.
            Y_data (np.array):                          Values to predict from input data.
            layers (int):                               Number of hidden layers
            activation_function (_type_):               Activation function to be used in hidden layers.
            activation_function_derivative (_type_):    Derivate of activation function.
            hidden_neurons (int, optional):             Number of neurons in each hidden layer. Defaults to 50.
            n_categories (int, optional):               Number of categories to predict. Defaults to 10.
            epochs (int, optional):                     Number of training epochs to run. Defaults to 10.
            batch_size (int, optional):                 Size of each batch in each epoch. Defaults to 100.
            learning_rate (float, optional):            Learning rate used in training. Defaults to 0.1.
            lam (float, optional):                      Regularization parameter. Defaults to 0.0.
            softmax_last_layer (bool, optional):        Decides if soft-max is used on ouput. Defaults to False.
        """
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.layers = layers
        self.softmax_last_layer = softmax_last_layer

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]

        self.hidden_neurons = hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.learning_rate = learning_rate
        self.lam = lam

        self.create_biases_and_weights()

    def create_biases_and_weights(self) -> None:
        """Generates biases and weights for each layers"""
        self.weights = [np.random.randn(self.n_features, self.hidden_neurons)]

        for i in range(self.layers - 1):
            self.weights.append(
                np.random.randn(self.hidden_neurons, self.hidden_neurons)
            )

        self.weights.append(np.random.randn(self.hidden_neurons, self.n_categories))

        self.bias = [np.zeros(self.hidden_neurons) + 0.01 for i in range(self.layers)]
        self.bias.append(np.zeros(self.n_categories) + 0.01)

        self.gradient_bias = [np.zeros(b.shape) for b in self.bias]
        self.gradient_w = [np.zeros(w.shape) for w in self.weights]

        # We need to add one, as we have an output layer
        self.layers += 1

    def feed_forward(self) -> None:
        """Perform a feed forward iteration, resulting in a prediction."""
        self.a_h = self.X_data

        self.activations = [self.a_h]
        self.zs = []

        for i in range(self.layers - 1):
            self.z_h = np.dot(self.a_h, self.weights[i]) + self.bias[i]
            self.a_h = self.activation_function(self.z_h)

            self.zs.append(self.z_h)
            self.activations.append(self.a_h)

        self.z_o = np.dot(self.a_h, self.weights[-1]) + self.bias[-1]

        if self.softmax_last_layer:
            exp_term = np.exp(self.z_o)
            self.z_o = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        self.zs.append(self.z_o)

    def feed_forward_out(self, X: np.array) -> np.array:
        """Predicts inputs in X, using a feed forward iteration.

        Args:
            X (np.array): Values to predict using the network.

        Returns:
            np.array: Predicted values
        """
        self.a_h = X

        for i in range(self.layers - 1):
            self.z_h = np.dot(self.a_h, self.weights[i]) + self.bias[i]
            self.a_h = self.activation_function(self.z_h)

        self.z_o = np.dot(self.a_h, self.weights[-1]) + self.bias[-1]

        if self.softmax_last_layer:
            exp_term = np.exp(self.z_o)
            self.z_o = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        return self.z_o

    def backpropagation(self) -> None:
        """Performs one iteration of backpropagation, updating the weights in the network."""
        error_output = self.z_o - self.Y_data

        delta = error_output

        self.gradient_bias[-1] = np.sum(delta, axis=0)
        self.gradient_w[-1] = np.dot(self.activations[-1].T, delta)

        if self.lam > 0.0:
            self.gradient_w[-1] += self.lam * self.weights[-1]

        for l in range(2, self.layers + 1):
            # print(l, len(self.activations))
            z = self.zs[-l]
            sp = self.activation_function_derivative(z)

            delta = np.dot(delta, self.weights[-l + 1].T) * sp

            self.gradient_bias[-l] = np.sum(delta, axis=0)
            # print(delta.shape)
            self.gradient_w[-l] = np.dot(self.activations[-l].T, delta)

            if self.lam > 0.0:
                self.gradient_w[-l] += self.lam * self.weights[-l]

        for i in range(self.layers):
            # print(i, self.weights[1].shape, self.gradient_w[-2].shape)
            self.weights[i] -= self.learning_rate * self.gradient_w[i]
            self.bias[i] -= self.learning_rate * self.gradient_bias[i]

    def predict(self, X: np.array) -> np.array:
        """Predicts inputs in X, using a feed forward iteration.
           And applying one-hot encoding if soft-max is used on output layer.

        Args:
            X (np.array): Values to predict using the network.

        Returns:
            np.array: Predicted values
        """
        y = self.feed_forward_out(X)

        if self.softmax_last_layer:
            y = np.argmax(y, axis=1)
            b = np.zeros((len(y), self.n_categories))
            b[np.arange(len(y)), y] = 1
            # print(b)
            # print(np.eye(len(y))[np.argmax(y, axis=1)])
            return b

        return y

    def train(self) -> None:
        """Performs the training of the network using SGD and backpropagation."""

        data_indices = np.arange(self.n_inputs)

        for i in tqdm(range(self.epochs), desc="Training"):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                # print(self.X_data)
                self.feed_forward()
                self.backpropagation()


if __name__ == "__main__":
    epochs = 2000
    batch_size = 20

    """ n = 20
    degree = 5
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    nr_of_params = (degree + 1) * (degree + 2) // 2
    X = np.zeros((x.size, nr_of_params), dtype=np.float64)
    idx = 0

    for j in range(0, degree + 1):
        for i in range(0, degree + 1):
            if i + j > degree:
                break
            X[:, idx] = np.power(x.flatten(), i) * np.power(y.flatten(), j)
            idx += 1

    input = (z.flatten()).reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, input, test_size=0.2)

    dnn = FFNN(
        X_data=X_train,
        Y_data=Y_train,
        activation_function=sigmoid,
        activation_function_derivative=deriv_sigmoid,
        learning_rate=0.001,
        lam=0.01,
        layers=2,
        epochs=epochs,
        batch_size=batch_size,
        hidden_neurons=50,
        n_categories=1,
        softmax_last_layer=False,
    )
    dnn.train()
    test_predict = dnn.predict(X_test)
    print(test_predict.shape)
    plt.scatter(X_train, test_predict)
    plt.scatter(X_train, Y_train)
    plt.show()
    # print(test_predict, Y_test)
    # accuracy score from scikit library
    print("Accuracy score on test set: ", mean_squared_error(Y_test, test_predict)) """

    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
    X = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets

    y = np.nan_to_num(y.to_numpy())
    X = np.nan_to_num(X.to_numpy())

    y = (y - 2) / 2
    y = np.array([(1 - i, i) for i in y]).reshape(X.shape[0], 2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    dnn = FFNN(
        X_data=X_train,
        Y_data=Y_train,
        activation_function=sigmoid,
        activation_function_derivative=deriv_sigmoid,
        learning_rate=0.0001,
        lam=0.01,
        layers=2,
        epochs=epochs,
        batch_size=batch_size,
        hidden_neurons=50,
        n_categories=2,
        softmax_last_layer=True,
    )

    dnn.train()
    test_predict = dnn.predict(X_test)
    print(test_predict.dtype, Y_test.dtype)

    print("Accuracy score on test set: ", accuracy_score(Y_test, test_predict))

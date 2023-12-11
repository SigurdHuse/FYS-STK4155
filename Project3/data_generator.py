from PINN import PINN
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import time
from finite_diff import Poisson2D
import numpy as np
import torchphysics as tp
import torch
import os
import torch.nn as nn

dir_name = "generated_data"


def test_function_1_numpy(x: np.array, y: np.array) -> np.array:
    """Computes manufactured solution using numpy arrays.

    Args:
        x (np.array): Values along x-axis.
        y (np.array): Values along y-axis.

    Returns:
        np.array: Computed manufactured solution.
    """
    return x*(1-x)*y*(1-y)*np.exp(np.sin(x) * np.cos(y))


def test_function_1(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """Computes manufactured solution.

    Args:
        x (torch.tensor): Points along x-axis.
        y (torch.tensor): Points along y-axis.

    Returns:
        torch.tensor: Computed manufactured solution.
    """
    return x*(1-x)*y*(1-y)*torch.exp(torch.sin(x) * torch.cos(y))


def plot_fn(u: torch.tensor, x: torch.tensor) -> torch.tensor:
    """Function to plot absolute difference between manufactured solution and predicted values.

    Args:
        u (torch.tensor): Values from network.
        x (torch.tensor): Coordinates of values.

    Returns:
        torch.tensor: Absolute difference between exact and predicted values.
    """
    exact = test_function_1(x[:, :1], x[:, 1:])
    return torch.abs(u - exact)


def laplace_test_function_1(x: torch.tensor) -> torch.tensor:
    """Computes laplace of manufactured solution.

    Args:
        x (torch.tensor): Computed values from network.

    Returns:
        torch.tensor: Laplace of manufactured solution.
    """
    xx = x[:, :1]
    y = x[:, 1:]
    cos_x, sin_x = torch.cos(xx), torch.sin(xx)
    cos_y, sin_y = torch.cos(y), torch.sin(y)

    f_xx = -y*(y - 1)*torch.exp(cos_y*sin_x)*(sin_x*cos_y*xx*xx - sin_x*cos_y*xx - cos_x*cos_x *
                                              cos_y*cos_y*xx*xx + cos_x*cos_x*cos_y*cos_y*xx - 4*cos_x*cos_y*xx + 2*cos_x*cos_y - 2)

    f_yy = -xx*(xx - 1)*torch.exp(cos_y*sin_x)*(cos_y * y*y * sin_x - cos_y*y*sin_x - y*y*sin_x *
                                                sin_x*sin_y*sin_y + y*sin_x*sin_x*sin_y*sin_y + 4*y*sin_x*sin_y - 2*sin_x*sin_y - 2)

    res = f_xx + f_yy
    return res


def bound_residual_1(u: torch.tensor, x: torch.tensor) -> torch.tensor:
    """Bound residual from poisson equation in 2D using manufactured solution.

    Args:
        u (torch.tensor): Values from network.
        x (torch.tensor): Dummy variable to fit specified syntax.

    Returns:
        torch.tensor: Bound residual.
    """
    return u


def pde_residual_1(u: torch.tensor, x: torch.tensor) -> torch.tensor:
    """PDE residual from poisson equation in 2D using manufactured solution.

    Args:
        u (torch.tensor): Values from network.
        x (torch.tensor): Variable to derivate u with respect to.

    Returns:
        torch.tensor: Computed PDE residual
    """
    return tp.utils.laplacian(u, x) - laplace_test_function_1(x)


def MSE(x_pred: np.array, x_true: np.array) -> float:
    """Computes mean squared error (MSE).

    Args:
        x_pred (np.array): Predicted values.
        x_true (np.array): Truth values

    Returns:
        float: Computed MSE.
    """
    return np.sum((x_pred - x_true)**2) / x_true.size


def compute_MSE_activation_function_reg_training(pde_residual, bound_residual, nr_sample_points_boundary: int, nr_sample_points_pde: int, f_function, test_function, activation_functions, reg_params, training_iterations, filename: str) -> None:
    """Computes MSE for different activation function, regularisation and epochs.

    Args:
        pde_residual (function):                Function describing PDE residiual.
        bound_residual (function):              Function describing boundary residual of PDE.
        nr_sample_points_boundary (int):        Number of points to sample along boundary.
        nr_sample_points_pde (int):             Number of points to sample in interior.
        f_function (function):                  Function describing right hand side of equation.
        test_function (function):               Test function used to compute MSE.
        activation_functions (list[function]):  List of activation functions to test.
        reg_params (list[float]):               List with regularisation parameters.
        training_iterations (list[int]):        List with number of training iterations.
        filename (str):                         Filename to save results as.
    """
    results = np.zeros((len(activation_functions), len(
        reg_params), len(training_iterations)))

    timings = np.zeros((len(activation_functions), len(
        reg_params), len(training_iterations)))

    for i, activation_function in enumerate(activation_functions):
        for j, reg_param in enumerate(reg_params):
            model = PINN(pde_residual=pde_residual, bound_residual=bound_residual,  nr_sample_points_boundary=nr_sample_points_boundary,
                         sample_points_pde=nr_sample_points_pde, boundary_weight=10, f=f_function, device_number=2, activation_function=activation_function)
            for k, iterations in enumerate(training_iterations):
                start = time.time()
                model.train("ADAM", 0.001, iterations, reg_param)
                end = time.time()
                cord, values, _ = model.extract_points(
                    10000, device="cpu")

                timings[i][j][k] = end - start

                f = test_function(cord[:, :1], cord[:, 1:])
                results[i][j][k] = MSE(values, f)

    np.savetxt(dir_name + "/" + filename,
               results.reshape(results.shape[0], -1))

    np.savetxt(dir_name + "/timings_" + filename,
               timings.reshape(timings.shape[0], -1))


def compute_MSE_finite_diff(N_vals: list[int], ue, name: str) -> None:
    """Computes MSE using finite difference for different mesh sizes.

    Args:
        N_vals (list[int]):     N-values to use for mesh size.
        ue (_type_):            Exact solution of PDE.
        name (str):             Filename to save results as.
    """
    model = Poisson2D(1.0, ue)
    results = np.zeros(len(N_vals))
    timings = np.zeros(len(N_vals))

    x, y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)

    xij, yij = np.meshgrid(x, y)
    exact = test_function_1_numpy(xij, yij)

    for i, N in enumerate(N_vals):
        start = time.time()
        model(N)
        end = time.time()
        timings[i] = end - start

        s = 0
        for cur_x, cur_y, f in zip(xij.ravel(), yij.ravel(), exact.ravel()):
            s += (model.eval(cur_x, cur_y) - f)**2

        results[i] = s / exact.size

    np.savetxt(dir_name + "/" + name, results)
    np.savetxt(dir_name + "/timings_" + name, timings)


def compute_MSE_for_different_architecture(nodes_in_each_layer: list[int], nr_of_layers: list[int], pde_residual, bound_residual, nr_sample_points_boundary: int, nr_sample_points_pde: int, f_function, test_function, training_iterations: int, filename: str) -> None:
    """Computes MSE for different network architectures using tanh activation function.

    Args:
        nodes_in_each_layer (list[int]):    List with how many nodes to have in each hidden layer.
        nr_of_layers (list[int]):           List of number of hidden layers.
        pde_residual (function):            Function describing PDE residiual.
        bound_residual (function)):         Function describing boundary residual of PDE.
        nr_sample_points_boundary (int):    Number of points to sample along boundary.
        nr_sample_points_pde (int):         Number of points to sample in interior.
        f_function (function):              Function describing right hand side of equation.
        test_function (function):           Test function used to compute MSE.
        training_iterations (int):          Number of epochs to train for.
        filename (str):                     Filename to save results as.                       
    """
    results = np.zeros((len(nodes_in_each_layer), len(nr_of_layers)))
    timings = np.zeros((len(nodes_in_each_layer), len(nr_of_layers)))

    for i, nodes in enumerate(nodes_in_each_layer):
        for j, layers in enumerate(nr_of_layers):
            cur = tuple([nodes]*layers)
            model = PINN(pde_residual=pde_residual, bound_residual=bound_residual,  nr_sample_points_boundary=nr_sample_points_boundary,
                         sample_points_pde=nr_sample_points_pde, boundary_weight=10, f=f_function, device_number=1, activation_function=nn.Tanh(), hidden_layers=cur)

            start = time.time()
            model.train("ADAM", 0.001,
                        max_steps=training_iterations, reg_param=1e-2)
            end = time.time()
            cord, values, _ = model.extract_points(
                10000, device="cpu")

            f = test_function(cord[:, :1], cord[:, 1:])
            results[i][j] = MSE(values, f)
            timings[i][j] = end - start

    np.savetxt(dir_name + "/" + filename, results)
    np.savetxt(dir_name + "/timings_" + filename, timings)


def long_run_of_big_network(nodes_in_each_layer: int, nr_of_layers: int, pde_residual, bound_residual, nr_sample_points_boundary: int, nr_sample_points_pde: int, f_function, test_function, nr_of_training_runs: int, training_iterations: int, filename: str, activation_function):
    """Trains specified network, saving MSE after each iteration.

    Args:
        nodes_in_each_layer (int):          Nodes in each hidden layer.
        nr_of_layers (int):                 Number of hidden layers.
        pde_residual (function):            Function describing PDE residiual.
        bound_residual (function):          Function describing boundary residual of PDE.
        nr_sample_points_boundary (int):    Number of points to sample along boundary.
        nr_sample_points_pde (int):         Number of points to sample in interior.
        f_function (function):              Function describing right hand side of equation.
        test_function (function):           Test function used to compute MSE.
        nr_of_training_runs (int):          Number of iterations.
        training_iterations (int):          Number of epochs each iteration.
        filename (str):                     Filename to save results as. 
        activation_function (function):     Specifies activation function to use.                    
    """

    cur = tuple([nodes_in_each_layer]*nr_of_layers)
    model = PINN(pde_residual=pde_residual, bound_residual=bound_residual, nr_sample_points_boundary=nr_sample_points_boundary,
                 sample_points_pde=nr_sample_points_pde, boundary_weight=10, f=f_function, device_number=2, activation_function=activation_function, hidden_layers=cur)

    results = np.zeros(nr_of_training_runs)
    timings = np.zeros(1)

    start = time.time()
    for i in range(nr_of_training_runs):
        model.train("ADAM", 0.001, training_iterations, reg_param=1e-2)
        cord, values, _ = model.extract_points(
            10000, device="cpu")

        f = test_function(cord[:, :1], cord[:, 1:])
        results[i] = MSE(f, values)
        # print(results[i], "\n\n\n")

    end = time.time()
    timings[0] = end - start

    np.savetxt(dir_name + "/" + filename, results)
    np.savetxt(dir_name + "/timings_" + filename, timings)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7" if torch.cuda.is_available(
    ) else "0"
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    """ compute_MSE_activation_function_reg_training(pde_residual=pde_residual_1, bound_residual=bound_residual_1, nr_sample_points_boundary=1000,
                                                 nr_sample_points_pde=6000, f_function=laplace_test_function_1, activation_functions=[nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()], reg_params=[0, 0.1, 0.2, 0.3, 0.4], training_iterations=[1000, 2000, 4000, 8000, 16000], filename="MSE_activation.bin", test_function=test_function_1_numpy) """

    compute_MSE_for_different_architecture(nodes_in_each_layer=[10, 20, 40, 80, 160], nr_of_layers=[1, 2, 4, 8, 12], pde_residual=pde_residual_1, bound_residual=bound_residual_1, nr_sample_points_boundary=1000,
                                           nr_sample_points_pde=6000, f_function=laplace_test_function_1, test_function=test_function_1_numpy, training_iterations=50_000, filename="MSE_networks.bin")

    long_run_of_big_network(nodes_in_each_layer=160, nr_of_layers=12, pde_residual=pde_residual_1, bound_residual=bound_residual_1, nr_sample_points_boundary=2000,
                            nr_sample_points_pde=12_000, f_function=laplace_test_function_1, test_function=test_function_1_numpy, nr_of_training_runs=200, training_iterations=1_000, filename="MSE_big_network_ReLU.bin", activation_function=nn.ReLU())

    long_run_of_big_network(nodes_in_each_layer=160, nr_of_layers=12, pde_residual=pde_residual_1, bound_residual=bound_residual_1, nr_sample_points_boundary=2000,
                            nr_sample_points_pde=12_000, f_function=laplace_test_function_1, test_function=test_function_1_numpy, nr_of_training_runs=200, training_iterations=1_000, filename="MSE_big_network_Tanh.bin", activation_function=nn.Tanh())

    """ compute_MSE_activation_function_reg_training(pde_residual=pde_residual_1, bound_residual=bound_residual_1, nr_sample_points_boundary=1000,
                                                 nr_sample_points_pde=6000, f_function=laplace_test_function_1, activation_functions=[nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()], reg_params=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], training_iterations=[1000, 2000, 4000, 8000, 16000], filename="MSE_activation_more.bin", test_function=test_function_1_numpy) """

    # ue = x*(1-x)*y*(1-y)*sp.exp(sp.sin(x) * sp.cos(y))
    # compute_MSE_finite_diff(
    #    N_vals=[10, 20, 40, 80, 160], ue=ue, name="MSE_finite_diff.bin")

    """ model = PINN(pde_residual=pde_residual_1, bound_residual=bound_residual_1, nr_sample_points_boundary=2000,
                 sample_points_pde=7500, boundary_weight=10, f=laplace_test_function_1, device_number=1)

    model.train("ADAM", 0.001, 20000, 0.1)
    # model.train("LFBGS", 0.05, 5000, 0.1)
    cord, values, domain = model.extract_points(1000, device="cpu")
    f = test_function_1_numpy(cord[:, :1], cord[:, 1:])

    print(np.sum((f - values)**2) / f.size)
    model.plot_figure(1000, lambda u: u, 'contour_surface', "test.png")
    model.plot_figure(1000, plot_fn, 'contour_surface', "error.png")

    xx, yy = np.linspace(0, 1, 50), np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(xx, yy)

    plt.matshow(test_function_1_numpy(xx, yy))
    plt.colorbar()
    plt.savefig("true.png") """

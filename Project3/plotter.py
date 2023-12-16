import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib.colors import SymLogNorm, LogNorm

from data_generator import test_function_1_numpy

dir_name = "plots"
# plt.rcParams["figure.figsize"] = [10, 10]
mpl.rcParams["figure.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 13
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["figure.dpi"] = 250


def compare_activation_functions(
    filename: str,
    time_steps: list,
    activation_functions: list,
    reg_params: list,
    name: str,
) -> None:
    """Creates plots comparing different activation functions, with respect to epochs and regularisation parameter.

    Args:
        filename (str):                     Filename to load data to plot from.
        time_steps (list[int]):             List with epochs used in data generation.
        activation_functions (list[str]):   List with names of different activation functions.
        reg_params (list[int]):             List with regularisation parameters used in data generation.
        name (str):                         Filename to save plot as.
    """
    # "generated_data/MSE_activation.bin"
    nr_of_steps = len(time_steps)
    input = np.loadtxt(filename)
    res = input.reshape(input.shape[0], input.shape[1] // nr_of_steps, nr_of_steps)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # plt.figure(figsize=(10, 10))
    # fig.set_dpi(figsize=(10, 10))

    # i, j = 0, 0

    for i, _ in enumerate(activation_functions):
        results = res[i]
        im = axs[i // 2, i % 2].matshow(
            results, cmap=cm.coolwarm, norm=LogNorm(vmin=np.min(res), vmax=np.max(res))
        )

        for (e, r), z in np.ndenumerate(results):
            axs[i // 2, i % 2].text(r, e, "{:.1e}".format(z), ha="center", va="center")

    for i in range(2):
        for j in range(2):
            axs[i, j].xaxis.set_major_locator(MultipleLocator(1))
            axs[i, j].yaxis.set_major_locator(MultipleLocator(1))
            axs[i, j].set_xticks(ticks=list(range(len(time_steps))))
            axs[i, j].set_xticklabels(
                labels=[f"{i : .1e}" for i in time_steps], rotation=90
            )
            axs[i, j].set_yticks(ticks=list(range(len(reg_params))))
            axs[i, j].set_yticklabels(
                labels=[f"{i : .1e}" for i in reg_params],
            )
            axs[i, j].xaxis.set_ticks_position("bottom")
            axs[i, j].set_ylabel(r"$\lambda$")
            axs[i, j].set_xlabel("epochs")
            axs[i, j].yaxis.set_label_position("right")

    axs[0, 0].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[0, 0].set_title(activation_functions[0])
    axs[0, 1].set_title(activation_functions[1])
    axs[1, 0].set_title(activation_functions[2])
    axs[1, 1].set_title(activation_functions[3])

    fig.colorbar(im, ax=axs.ravel().tolist(), label="Mean squared error")
    # fig.tight_layout()
    plt.savefig(dir_name + "/" + name + ".pdf")
    plt.close()


def compare_running_times(
    filename: str, time_steps: list, name: str, activation_functions: list
) -> None:
    """Create plot comparing running times of data generated.

    Args:
        filename (str):                     Filename to load data to plot from.
        time_steps (list[int]):             List with epochs used in data generation.
        name (str):                         Filename to save plot as.
        activation_functions (list[str]):   List with names of different activation functions.
    """
    nr_of_steps = len(time_steps)
    input = np.loadtxt(filename)
    res = input.reshape(input.shape[0], input.shape[1] // nr_of_steps, nr_of_steps)

    for i, activation_function in enumerate(activation_functions):
        cur = res[i]
        means = np.mean(cur, axis=0)
        stds = np.std(cur, axis=0)

        plt.errorbar(time_steps, means, yerr=stds, label=activation_function)

    plt.plot(time_steps, np.array(time_steps) / 1e2, label="O(N)")
    plt.grid()
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(dir_name + "/" + name)
    plt.close()
    print(res.shape)


def compare_networks(filename: str, layers: list, nodes: list, name: str) -> None:
    """Make plot comparing MSE for different network architectures.

    Args:
        filename (str):         Filename to load data to plot from.
        layers (list[int]):     List with number of hidden layers used in data generation.
        nodes (list[int]):      List with number of nodes in each hidden layers used in data generation.
        name (str):             Filename to save plot as.
    """

    res = np.loadtxt(filename)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    tmp = ax.matshow(
        res, cmap=cm.coolwarm, norm=LogNorm(vmin=np.min(res), vmax=np.max(res))
    )
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(ticks=list(range(len(layers))))
    ax.set_xticklabels(labels=[f"{i}" for i in layers])
    ax.set_yticks(ticks=list(range(len(nodes))))
    ax.set_yticklabels(
        labels=[f"{i}" for i in nodes],
    )
    # ax.xaxis.set_ticks_position("bottom")
    ax.set_ylabel("Nodes in each hidden layer")
    ax.set_xlabel("Number of layers")
    ax.yaxis.set_label_position("right")

    for (e, r), z in np.ndenumerate(res):
        ax.text(r, e, "{:.1e}".format(z), ha="center", va="center")

    fig.colorbar(tmp, label="Mean squared error")

    plt.savefig(dir_name + "/" + name)
    plt.figure()


def plot_manufactured_solution(filename: str, N: int) -> None:
    """Plot manufactured solution.

    Args:
        filename (str):     Filename to store plot as.
        N (int):            Number of data points in x- and y-direction.
    """
    x, y = np.linspace(0, 1, N), np.linspace(0, 1, N)
    xij, yij = np.meshgrid(x, y)

    tmp = plt.matshow(test_function_1_numpy(xij, yij), cmap=cm.coolwarm)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.xticks(
        ticks=np.linspace(0, N - 1, 11),
        labels=[f"{i : .1f}" for i in np.linspace(0, 1, 11)],
        rotation=90,
    )
    plt.yticks(
        ticks=np.linspace(0, N - 1, 11),
        labels=[f"{i : .1f}" for i in np.linspace(0, 1, 11)],
    )
    plt.colorbar(tmp, label=r"$u_{test}(x,y)$")
    plt.savefig(dir_name + "/" + filename)


def compute_average_run_time(filename1: str, filename2: str) -> None:
    """Computes the average run time.

    Args:
        filename1 (str): Filename of first running times.
        filename2 (str): Filename of second running times.
    """
    nr_of_steps = 5
    input1 = np.loadtxt(filename1)
    input2 = np.loadtxt(filename2)
    res1 = input1.reshape(input1.shape[0], input1.shape[1] // nr_of_steps, nr_of_steps)
    res2 = input1.reshape(input2.shape[0], input2.shape[1] // nr_of_steps, nr_of_steps)
    run_times = np.zeros((2 * nr_of_steps * res1.shape[0], nr_of_steps))

    for i in range(res1.shape[0]):
        run_times[nr_of_steps * i : nr_of_steps * (i + 1), :] = res1[i]
    offset = res1.shape[0] * nr_of_steps
    for i in range(res2.shape[0]):
        run_times[offset + nr_of_steps * i : offset + nr_of_steps * (i + 1), :] = res2[
            i
        ]
    run_times = np.cumsum(run_times, axis=1)
    print(np.mean(run_times, axis=0))
    print(np.std(run_times, axis=0))


def compare_nr_of_epohs(
    filename: str,
    name: str,
    epochs_each_iteration: int,
    activation_function: str,
    pad: float,
) -> None:
    res = np.loadtxt(filename)
    nr_of_steps = len(res)
    nr_of_epochs = nr_of_steps * epochs_each_iteration

    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(epochs_each_iteration, nr_of_epochs + 1, epochs_each_iteration),
        res,
        label=activation_function,
    )
    plt.tight_layout(pad=pad)
    plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()

    plt.savefig(dir_name + "/" + name)


def plot_MSE_finite_diff(filename: str, name: str, n_vals: np.array):
    data = np.loadtxt(filename)

    plt.plot(n_vals, data, label="MSE finite difference")
    plt.legend()
    plt.xlabel("N value")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.grid()
    plt.savefig(dir_name + "/" + name)


if __name__ == "__main__":
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    compare_activation_functions(
        filename="generated_data/MSE_activation.bin",
        time_steps=[1000, 3000, 7000, 15000, 31000],
        activation_functions=["ReLu", "Sigmoid", "Tanh", "Leaky ReLu"],
        reg_params=[2e-4, 5e-4, 5e-5, 1e-6, 0.0],
        name="compare_activation",
    )
    plt.clf()
    compare_activation_functions(
        filename="generated_data/MSE_activation_more.bin",
        time_steps=[1000, 3000, 7000, 15000, 31000],
        activation_functions=["ReLu", "Sigmoid", "Tanh", "Leaky ReLu"],
        reg_params=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        name="compare_activation_more",
    )
    plt.clf()
    compute_average_run_time(
        filename1="generated_data/timings_MSE_activation.bin",
        filename2="generated_data/timings_MSE_activation_more.bin",
    )

    compare_running_times(
        filename="generated_data/timings_MSE_activation.bin",
        time_steps=[1000, 2000, 4000, 8000, 16000],
        name="time_epochs",
        activation_functions=["ReLu", "Sigmoid", "Tanh", "LeakyReLu"],
    )
    plt.clf()
    compare_networks(
        filename="generated_data/MSE_networks.bin",
        layers=[1, 2, 4, 8, 12],
        nodes=[10, 20, 40, 80, 160],
        name="MSE_networks.pdf",
    )
    plt.clf()
    plot_manufactured_solution("manufactured_solution.png", 1000)

    plt.clf()
    compare_nr_of_epohs(
        "generated_data/MSE_big_network_Tanh_8_160.bin",
        "Tanh_long_run.pdf",
        1000,
        "Tanh",
        2.0,
    )
    plt.clf()
    compare_nr_of_epohs(
        "generated_data/MSE_big_network_ReLU_8_160.bin",
        "ReLu_long_run.pdf",
        1000,
        "ReLu",
        3.5,
    )
    plt.clf()
    plot_MSE_finite_diff(
        filename="generated_data/MSE_finite_diff.bin",
        name="finite_diff.pdf",
        n_vals=np.array([10, 20, 40, 80, 160]),
    )
    plt.clf()
    tmp = np.loadtxt("generated_data/timings_MSE_finite_diff.bin")
    for i in tmp:
        print(f"{i : f}")

    tmp = np.loadtxt("generated_data/timings_MSE_networks.bin")
    print(np.mean(tmp, axis=0))
    print(np.std(tmp, axis=0))

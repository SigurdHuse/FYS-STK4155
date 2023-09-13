from regression import OSLpredictor, Ridgepredictor, Lassopredictor, FrankeFunction
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

dir_name = "images"
seed = 2023


def plot_MSE_or_R2_as_func_of_degree(
    x: np.array,
    y: np.array,
    z: np.array,
    nr_of_degs: int,
    threshold: float,
    scale: bool,
    lam: list,
    filename: str,
    regression_type: str,
    plot_MSE: bool,
    plot_with_lam: bool,
) -> None:
    nr_of_degs = int(nr_of_degs)
    degrees = list(range(1, nr_of_degs + 1))
    results_test = np.zeros((len(lam), nr_of_degs))
    results_train = np.zeros((len(lam), nr_of_degs))

    for deg in degrees:
        if regression_type == "OLS":
            model = OSLpredictor(x, y, z, deg, threshold, scale)
        elif regression_type == "Ridge":
            model = Ridgepredictor(x, y, z, deg, threshold, scale)
        elif regression_type == "Lasso":
            model = Lassopredictor(x, y, z, deg, threshold, scale)
        else:
            raise NotImplementedError(
                f"The {regression_type} regression is not implemented"
            )

        for idx, la in enumerate(lam):
            model.compute_parameters(la)
            model.predict_test()
            model.predict_train()

            if plot_MSE:
                results_train[idx, deg - 1] = model.MSE(on_training=True)
                results_test[idx, deg - 1] = model.MSE(on_training=False)
            else:
                results_train[idx, deg - 1] = model.R2(on_training=True)
                results_test[idx, deg - 1] = model.R2(on_training=False)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for idx, la in enumerate(lam):
        if plot_MSE:
            label = "MSE"
            if plot_with_lam:
                label += rf", $\lambda = $ {lam[idx]}"
        else:
            label = "R2"
            if plot_with_lam:
                label += rf", $\lambda = $ {lam[idx]}"

        ax1.plot(degrees, results_train[idx], label=label)
        ax2.plot(degrees, results_test[idx], label=label)

    if plot_MSE:
        ax1.set(ylabel="MSE", xlabel="Degree")
        ax2.set(ylabel="MSE", xlabel="Degree")
    else:
        ax1.set(ylabel="R2-score", xlabel="Degree")
        ax2.set(ylabel="R2-score", xlabel="Degree")

    ax1.set(title="Training data")
    ax2.set(title="Test data")

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.savefig(dir_name + "/" + filename)


if __name__ == "__main__":
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    np.random.seed(seed)
    n = 100

    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    plot_MSE_or_R2_as_func_of_degree(
        x.flatten(),
        y.flatten(),
        z.flatten(),
        12,
        0.2,
        True,
        [1, 0.1, 0.001],
        "MSE_OSL.png",
        "Ridg",
        True,
        True,
    )

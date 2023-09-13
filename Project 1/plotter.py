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
    """Plots the MSE or R2 as a function of maximum degree of polynomials in design matrix.

    Args:
        x (np.array): x-coordinates.
        y (np.array): y-coordinates.
        z (np.array): Values to predict, also z-coordinates.
        nr_of_degs (int): Plots the degrees up to this degree.
        threshold (float): Treshold used to split data.
        scale (bool): Determines if data is scaled.
        lam (list): List of lambda values to compute MSE and R2 for.
        filename (str): Filename to save plot as.
        regression_type (str): Which type of regression to be used.
                               It is only possible to use OLS, Ridge and Lasso.
        plot_MSE (bool): True if MSE is to be plotted, False if R2 should be plotted.
        plot_with_lam (bool): True if lambda values are to be part of label.

    Raises:
        NotImplementedError: Raised if a regression type which is not implmented is passed as regression_type.
    """
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

    fig.suptitle(f"Regression using {regression_type}")
    ax1.set(title="Training data")
    ax2.set(title="Test data")

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.savefig(dir_name + "/" + filename)


def plot_MSE_using_bootstrap_and_OSL(nr_of_its, nr_of_degs) -> None:
    pass


if __name__ == "__main__":
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    np.random.seed(seed)
    n = 100

    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    osl_plots = [(True, "MSE_OSL.png"), (False, "R2_OSL.png")]

    for i in osl_plots:
        plot_MSE_or_R2_as_func_of_degree(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            nr_of_degs=10,
            threshold=0.2,
            scale=True,
            lam=[0],
            filename=i[1],
            regression_type="OLS",
            plot_MSE=i[0],
            plot_with_lam=False,
        )
    lambdas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    ridge_plots = [(True, "MSE_Ridge.png"), (False, "R2_Ridge.png")]
    for i in ridge_plots:
        plot_MSE_or_R2_as_func_of_degree(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            nr_of_degs=10,
            threshold=0.2,
            scale=True,
            lam=lambdas,
            filename=i[1],
            regression_type="Ridge",
            plot_MSE=i[0],
            plot_with_lam=True,
        )

    alphas = [0.1, 0.01, 0.001, 0.0001]
    lasso_plots = [(True, "MSE_Lasso.png"), (False, "R2_Lasso.png")]
    for i in lasso_plots:
        plot_MSE_or_R2_as_func_of_degree(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            nr_of_degs=10,
            threshold=0.2,
            scale=True,
            lam=alphas,
            filename=i[1],
            regression_type="Lasso",
            plot_MSE=i[0],
            plot_with_lam=True,
        )

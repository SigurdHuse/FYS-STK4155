from regression import OLSpredictor, Ridgepredictor, Lassopredictor, FrankeFunction
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
    variance_of_noise: float,
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

    for j, deg in enumerate(degrees):
        if regression_type == "OLS":
            model = OLSpredictor(x, y, z, deg, threshold, scale, variance_of_noise)
        elif regression_type == "Ridge":
            model = Ridgepredictor(x, y, z, deg, threshold, scale, variance_of_noise)
        elif regression_type == "Lasso":
            model = Lassopredictor(x, y, z, deg, threshold, scale, variance_of_noise)
        else:
            raise NotImplementedError(
                f"The {regression_type} regression is not implemented"
            )
        # print(model.X.dtype)
        for idx, la in enumerate(lam):
            model.compute_parameters(la)
            model.predict_test()
            model.predict_train()

            if plot_MSE:
                results_train[idx, j] = model.MSE(on_training=True)
                results_test[idx, j] = model.MSE(on_training=False)
            else:
                results_train[idx, j] = model.R2(on_training=True)
                results_test[idx, j] = model.R2(on_training=False)

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
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.grid()
    ax2.grid()
    ax1.legend()
    # ax2.legend()
    plt.tight_layout()
    plt.savefig(dir_name + "/" + filename)


def plot_Bias_variance_using_bootstrap_and_OLS(
    x: np.array,
    y: np.array,
    z: np.array,
    nr_of_its: int,
    mx_degree: int,
    treshold: float,
    scale: bool,
    filename: str,
    variance_of_noise: float,
) -> None:
    degrees = list(range(1, mx_degree + 1))
    error = np.zeros(mx_degree)
    bias = np.zeros(mx_degree)
    variance = np.zeros(mx_degree)

    for i, deg in enumerate(degrees):
        model = OLSpredictor(x, y, z, deg, treshold, scale, variance_of_noise)
        model.bootstrap(nr_of_its, None)

        model.z_test = model.z_test.reshape(-1, 1)
        # print(model.bootstrap_results)
        error[i] = np.mean(
            np.mean(
                (model.z_test - model.bootstrap_results) ** 2, axis=1, keepdims=True
            )
        )

        bias[i] = np.mean(
            (model.z_test - np.mean(model.bootstrap_results, axis=1, keepdims=True))
            ** 2
        )
        variance[i] = np.mean(np.var(model.bootstrap_results, axis=1, keepdims=True))

        # print(model.bootstrap_results)
    plt.plot(degrees, bias, label="Bias", linestyle=":", linewidth=3)
    plt.plot(degrees, error, label="Error")
    plt.plot(degrees, variance, label="Variance")

    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "/" + filename)


def plot_bias_variance_using_cross_validation(
    x: np.array,
    y: np.array,
    z: np.array,
    nr_of_groups: list,
    mx_degree: int,
    lam: list,
    model: str,
    threshold: float,
    scale: bool,
    filename: str,
    variance_of_noise: float,
) -> None:
    degrees = list(range(1, mx_degree + 1))
    results = np.zeros((len(nr_of_groups), mx_degree))

    for deg in degrees:
        for idx, group in enumerate(nr_of_groups):
            if model == "OLS":
                cur_model = OLSpredictor(
                    x, y, z, deg, threshold, scale, variance_of_noise
                )
            elif model == "Ridge":
                cur_model = Ridgepredictor(
                    x, y, z, deg, threshold, scale, variance_of_noise
                )
            elif model == "Lasso":
                cur_model = Lassopredictor(
                    x, y, z, deg, threshold, scale, variance_of_noise
                )
            else:
                raise NotImplementedError

            cur_model.cross_validation(group, lam)
            results[idx, deg - 1] = np.mean(cur_model.results_cross_val)

    for i, k in enumerate(nr_of_groups):
        plt.plot(degrees, results[i], label=f"Error, k = {k}")

    plt.xlabel("Degree")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "/" + filename)


if __name__ == "__main__":
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    plt.rc("pgf", texsystem="pdflatex")
    variance = 0.01
    np.random.seed(seed)
    n = 20

    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    ols_plots = [(True, "MSE_OLS.pgf"), (False, "R2_OLS.pgf")]

    for i in ols_plots:
        plot_MSE_or_R2_as_func_of_degree(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            nr_of_degs=20,
            threshold=0.2,
            scale=True,
            lam=[0],
            filename=i[1],
            regression_type="OLS",
            plot_MSE=i[0],
            plot_with_lam=False,
            variance_of_noise=variance,
        )
        plt.clf()
    lambdas = [10, 1, 0.1, 0.01, 0.001]
    ridge_plots = [(True, "MSE_Ridge.pgf"), (False, "R2_Ridge.pgf")]
    for i in ridge_plots:
        plot_MSE_or_R2_as_func_of_degree(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            nr_of_degs=20,
            threshold=0.2,
            scale=True,
            lam=lambdas,
            filename=i[1],
            regression_type="Ridge",
            plot_MSE=i[0],
            plot_with_lam=True,
            variance_of_noise=variance,
        )
        plt.clf()

    alphas = [10, 1, 0.1, 0.01, 0.001]
    lasso_plots = [(True, "MSE_Lasso.pgf"), (False, "R2_Lasso.pgf")]
    for i in lasso_plots:
        plot_MSE_or_R2_as_func_of_degree(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            nr_of_degs=20,
            threshold=0.2,
            scale=True,
            lam=alphas,
            filename=i[1],
            regression_type="Lasso",
            plot_MSE=i[0],
            plot_with_lam=True,
            variance_of_noise=variance,
        )
        plt.clf()

    """ plot_Bias_variance_using_bootstrap_and_OLS(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        nr_of_its=1000,
        mx_degree=5,
        treshold=0.2,
        scale=True,
        filename="Bias_variance_bootstrap.pgf",
        variance_of_noise=variance,
    ) """
    """ plt.clf()
    plot_bias_variance_using_cross_validation(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        nr_of_groups=[5, 6, 7, 8, 9, 10],
        mx_degree=5,
        lam=0.01,
        model="OLS",
        threshold=0.2,
        scale=True,
        filename="cross_validation_OLS.pgf",
        variance_of_noise=variance,
    ) """

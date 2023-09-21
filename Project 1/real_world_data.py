import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from regression import OLSpredictor, Ridgepredictor, Lassopredictor


if __name__ == "__main__":
    # Load the terrain
    terrain1 = imread("data/SRTM_data_Norway_1.tif")

    terrain1 = terrain1[::100, ::100]
    y = np.arange(0, terrain1.shape[0])
    x = np.arange(0, terrain1.shape[1])
    x, y = np.meshgrid(x, y)

    """ fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        x,
        y,
        terrain1,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.show() """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.tight_layout()
    degrees = list(range(3, 10 + 1))
    results_OLS = np.zeros(len(degrees))
    for idx, deg in enumerate(degrees):
        OLS_model = OLSpredictor(
            x.flatten(), y.flatten(), terrain1.flatten(), deg, 0.2, True
        )
        OLS_model.compute_parameters()
        OLS_model.cross_validation(10, None)
        results_OLS[idx] = np.mean(OLS_model.results_cross_val)

    ax1.plot(degrees, results_OLS)
    lambdas = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    results_Ridge = np.zeros((len(lambdas), len(degrees)))

    for j, deg in enumerate(degrees):
        Ridge_model = Ridgepredictor(
            x.flatten(), y.flatten(), terrain1.flatten(), deg, 0.2, True
        )
        for i, lam in enumerate(lambdas):
            Ridge_model.cross_validation(10, lam)
            results_Ridge[i, j] = np.mean(Ridge_model.results_cross_val)

    for i in range(len(lambdas)):
        ax2.plot(degrees, results_Ridge[i])

    alphas = [10, 1, 0.1, 0.01, 0.001]
    results_Lasso = np.zeros((len(alphas), len(degrees)))

    """ for j, deg in enumerate(degrees):
        Lasso_model = Lassopredictor(
            x.flatten(), y.flatten(), terrain1.flatten(), deg, 0.2, True
        )
        for i, alpha in enumerate(alphas):
            Lasso_model.cross_validation(10, alpha)
            results_Lasso[i, j] = np.mean(Lasso_model.results_cross_val)

    for i in range(len(alphas)):
        ax3.plot(degrees, results_Lasso[i]) """

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    plt.show()

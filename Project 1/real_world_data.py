import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from regression import OLSpredictor, Ridgepredictor, Lassopredictor
import os
from plotter import (
    plot_MSE_and_R2_as_func_of_degree,
    plot_Bias_variance_using_bootstrap_and_OLS,
    plot_bias_variance_using_cross_validation,
    plot_entire_dataset,
)

dir_name = "plots_real_data"
data_dir_name = "data_real_data"
file_extension = ".pgf"
seed = 2023


def make_3d_comparison(
    x: np.array,
    y: np.array,
    z: np.array,
    degree: int,
    threshold: float,
    scale: bool,
    lam: float,
    filename: str,
    regression_type: str,
    variance_of_noise: float,
    dir_name: str,
):
    """Plots the predicted model, and a heatmap showing absolute difference to real data.

    Args:
        x (np.array): x-coordinates.
        y (np.array): y-coordinates.
        z (np.array): Values to predict (z-coordinates).
        degree (int): Degree of polynomial to be used in design matrix.
        threshold (float): Treshold for splitting z into test and training data.
        scale (bool): Determines of design matrix is to be scaled.
        lam (float):  Lambda paramter to be used in model.
        filename (str): Filename to store plots as.
        regression_type (str): Determines regression model used to predict z.
        variance_of_noise (float): Variance of noise to be added to z.
        dir_name (str): Directory name to save plots in.

    Raises:
        NotImplementedError: If regression_type is not implemented.
    """
    h, w = z.shape
    if regression_type == "OLS":
        model = OLSpredictor(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            degree,
            threshold,
            scale,
            variance_of_noise,
        )
    elif regression_type == "Ridge":
        model = Ridgepredictor(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            degree,
            threshold,
            scale,
            variance_of_noise,
        )
    elif regression_type == "Lasso":
        model = Lassopredictor(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            degree,
            threshold,
            scale,
            variance_of_noise,
        )
    else:
        raise NotImplementedError(
            f"The {regression_type} regression is not implemented"
        )

    model.compute_parameters(lam)
    prediction = model.predict_entire_dataset().reshape(h, w)

    fig = plt.figure(dpi=250)

    # ax1.set(title="Real data")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        x,
        y,
        prediction,
        rstride=1,
        cstride=1,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    # ax2.set(title = "Predicted")
    ax2 = fig.add_subplot(1, 2, 2)
    surf = ax2.matshow(prediction)

    fig.colorbar(surf, label="Height [m]")

    # ax3.axes.get_xaxis().set_ticks([])
    # ax3.axes.get_yaxis().set_ticks([])

    fig.tight_layout(pad=3.5)
    plt.savefig(dir_name + "/" + filename)
    plt.close()

    fig = plt.figure(dpi=250)
    fig.tight_layout(pad=3.5)
    ax1 = fig.add_subplot(1, 1, 1)
    surf = ax1.matshow(np.abs(z - prediction))
    fig.colorbar(surf, label="Absolute difference between predicted and real data [m]")
    plt.savefig(dir_name + "/comparison_" + filename)


if __name__ == "__main__":
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    if not os.path.isdir(data_dir_name):
        os.mkdir(data_dir_name)

    np.random.seed(seed)
    terrain1 = imread("data/SRTM_data_Norway_1.tif")

    # terrain1 = terrain1[::10, ::10]
    y = np.arange(0, terrain1.shape[0])
    x = np.arange(0, terrain1.shape[1])
    x, y = np.meshgrid(x, y)

    make_3d_comparison(
        x=x,
        y=y,
        z=terrain1,
        degree=6,
        threshold=0.2,
        scale=True,
        lam=1e-4,
        filename="3D_comp.jpg",
        regression_type="OLS",
        variance_of_noise=0.0,
        dir_name=dir_name,
    )

    plot_entire_dataset(
        x=x, y=y, z=terrain1, dir_name=dir_name, file_name="entire_dataset.jpg"
    )
    #
    #
    plots = [
        ([0], "real_data_OLS", "OLS", False),
        ([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], "real_data_Ridge", "Ridge", True),
        ([1, 0.1, 0.05, 0.01], "real_data_Lasso", "Lasso", True),
    ]

    for i in plots:
        plot_MSE_and_R2_as_func_of_degree(
            x=x.flatten(),
            y=y.flatten(),
            z=terrain1.flatten(),
            nr_of_degs=6,
            threshold=0.2,
            scale=True,
            lam=i[0],
            filename=i[1],
            regression_type=i[2],
            plot_with_lam=i[3],
            variance_of_noise=0.0,
            write_to_file=True,
            dir_name=dir_name,
            data_dir_name=data_dir_name,
            use_real_tick=False,
            file_extension=file_extension,
        )
        plt.close()

    small_terrain = terrain1[::10, ::10]
    print(terrain1.size, small_terrain.size)

    small_x, small_y = np.meshgrid(
        np.arange(0, small_terrain.shape[0]), np.arange(0, small_terrain.shape[1])
    )

    plot_Bias_variance_using_bootstrap_and_OLS(
        x=small_x.flatten(),
        y=small_y.flatten(),
        z=small_terrain.flatten(),
        nr_of_its=1_000,
        mx_degree=20,
        treshold=0.2,
        scale=True,
        filename="Bias_variance_real_data",
        variance_of_noise=0.0,
        dir_name=dir_name,
        file_extension=file_extension,
    )

    plt.close()
    plot_bias_variance_using_cross_validation(
        x=small_x.flatten(),
        y=small_y.flatten(),
        z=small_terrain.flatten(),
        nr_of_groups=[5, 6, 7, 8, 9, 10],
        mx_degree=40,
        lam=[0],
        model="OLS",
        threshold=0.2,
        scale=True,
        filename="Cross_validation_real",
        variance_of_noise=0.0,
        dir_name=dir_name,
        file_extension=".pdf",
    )

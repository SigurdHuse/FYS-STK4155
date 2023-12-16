import torch
import numpy as np
import torchphysics as tp
import pytorch_lightning as pl
import torch.nn as nn
import matplotlib.pyplot as plt


class PINN:
    """Physics informed neural network for solving possion equation in 2D, in [0,1]^2, with dirichlet boundary conditions."""

    def __init__(self, pde_residual, bound_residual, nr_sample_points_boundary: int, sample_points_pde: int,
                 boundary_weight: float, f, device_number: int = None,
                 hidden_layers: tuple = (60, 60, 60, 60, 60), activation_function=nn.Tanh()) -> None:
        """Constructor.

        Args:
            pde_residual (_type_):                  PDE residual, describing interior points.
            bound_residual (_type_):                Boundary residual.
            nr_sample_points_boundary (int):        Number of points to sample along boundary during training.
            sample_points_pde (int):                Number of points to sample of interior points during training.
            boundary_weight (float):                Weight to multiply boundary residual with.
            f (_type_):                             Right-hand side function of PDE.
            device_number (int, optional):          Which CUDA device to use during training. Defaults to None.
            hidden_layers (tuple, optional):        Tuple describing the hidden layers. Defaults to (60, 60, 60, 60, 60).
            activation_function (_type_, optional): Activation function to utilise during training. Defaults to nn.Tanh().
        """
        self.device_number = device_number
        self.pde_residual = pde_residual
        self.bound_residual = bound_residual
        self.f = f

        X = tp.spaces.R2('x')  # input is 2D
        U = tp.spaces.R1('u')  # output is 1D
        self.square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])

        self.model = tp.models.FCN(
            input_space=X, output_space=U, hidden=hidden_layers, activations=activation_function)

        self.bound_sampler = tp.samplers.GridSampler(
            self.square.boundary, n_points=nr_sample_points_boundary)
        self.bound_sampler = self.bound_sampler.make_static()

        self.bound_cond = tp.conditions.PINNCondition(module=self.model, sampler=self.bound_sampler,
                                                      residual_fn=self.bound_residual, weight=boundary_weight)

        self.pde_sampler = tp.samplers.RandomUniformSampler(
            self.square, n_points=sample_points_pde)

    def train(self, model: str, learning_rate: float, max_steps: int, reg_param: float) -> None:
        """Trains the model.

        Args:
            model (str):            Specifies which SGD method to utilise. Currently two options: ADAM and LFBGS
            learning_rate (float):  Learning rate to use during training.
            max_steps (int):        Number of epochs to train network for.
            reg_param (float):      Regularsiation parameter.

        Raises:
            NotImplementedError: If model other than ADAM or LFBGS is specified.
        """
        self.pde_cond = tp.conditions.PINNCondition(
            module=self.model, sampler=self.pde_sampler, residual_fn=self.pde_residual,  data_functions={'f': self.f})

        if model == "ADAM":
            optim = tp.OptimizerSetting(
                optimizer_class=torch.optim.Adam, lr=learning_rate, optimizer_args={"weight_decay": reg_param})
        elif model == "LFBGS":
            # LBFGS does not work with varying points!
            self.pde_cond.sampler = self.pde_cond.sampler.make_static()
            optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=learning_rate,
                                        optimizer_args={'max_iter': 2, 'history_size': 100})
        else:
            raise NotImplementedError(f"Model {model} is not implemented!")

        solver = tp.solver.Solver(
            train_conditions=[self.bound_cond, self.pde_cond], optimizer_setting=optim)

        trainer = pl.Trainer(accelerator="gpu", gpus=[self.device_number],
                             max_steps=max_steps,  # number of training steps
                             logger=False,
                             benchmark=True,
                             enable_checkpointing=False)

        trainer.fit(solver)

    def extract_points(self, nr_of_points: int, device: str = "cuda") -> [np.array, np.array, np.array]:
        """Exactracts randomly sampled predicted points in the domain.

        Args:
            nr_of_points (int):     Number of points to extract.
            device (str, optional): Which device to use for predicting points. Defaults to "cuda".

        Returns:
            [np.array, np.array, np.array]: Return coordinates, values and domain points of predicted points in domain.
        """
        plot_sampler = tp.samplers.PlotSampler(
            plot_domain=self.square, n_points=nr_of_points, device=device)

        inp_points, output, _ = tp.utils.plotting.plot_functions._create_plot_output(
            self.model, lambda u: u, plot_sampler, device=device)

        domain_points = tp.utils.plotting.plot_functions._extract_domain_points(inp_points, plot_sampler.domain,
                                                                                len(plot_sampler))

        coordinates = output["x"].detach().numpy()
        values = output["u"].detach().numpy()

        return coordinates, values, domain_points

    def extract_points_given_sampler(self, sampler, domain, device: str = "cuda"):
        """Exactracts randomly sampled predicted points in the domain, using a predetermined sampler."""
        inp_points, output, _ = tp.utils.plotting.plot_functions._create_plot_output(
            self.model, lambda u: u, sampler, device=device)

        domain_points = tp.utils.plotting.plot_functions._extract_domain_points(inp_points, domain,
                                                                                len(sampler))

        coordinates = output["x"].detach().numpy()
        values = output["u"].detach().numpy()

        return coordinates, values, domain_points

    def plot_figure(self, n_points: int, lambda_func, plot_type: str, filename: str):
        """Plots figure using randomly sampled predicted points.

        Args:
            n_points (int):         Number of points to sample.
            lambda_func (_type_):   Function describing what to plot.
            plot_type (str):        Specifies what kind of plot to create.
            filename (str):         Filename to save plot as.
        """
        plot_sampler = tp.samplers.PlotSampler(
            plot_domain=self.square, n_points=n_points, device='cuda')

        fig = tp.utils.plot(self.model, lambda_func, plot_sampler,
                            plot_type=plot_type)

        plt.savefig(filename)
        plt.close()

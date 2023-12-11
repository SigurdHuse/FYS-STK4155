import torch
import numpy as np
import torchphysics as tp
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt

X = tp.spaces.R2('x')  # input is 2D
U = tp.spaces.R1('u')  # output is 1D

square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])

model = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 50))


def bound_residual(u, x):
    bound_values = torch.sin(np.pi/2*x[:, :1]) * torch.cos(2*np.pi*x[:, 1:])
    return u - bound_values


# the point sampler, for the trainig points:
# here we use grid points any other sampler could also be used
bound_sampler = tp.samplers.GridSampler(square.boundary, n_points=5000)
bound_sampler = bound_sampler.make_static()


def pde_residual(u, x):
    return tp.utils.laplacian(u, x) + 4.25*np.pi**2*u


# the point sampler, for the trainig points:
pde_sampler = tp.samplers.GridSampler(
    square, n_points=15000)  # again point grid
pde_sampler = pde_sampler.make_static()

# wrap everything together in the condition
pde_cond = tp.conditions.PINNCondition(module=model, sampler=pde_sampler,
                                       residual_fn=pde_residual)


bound_cond = tp.conditions.PINNCondition(module=model, sampler=bound_sampler,
                                         residual_fn=bound_residual, weight=10)

optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)

solver = tp.solver.Solver(
    train_conditions=[bound_cond, pde_cond], optimizer_setting=optim)

os.environ["CUDA_VISIBLE_DEVICES"] = "1" if torch.cuda.is_available() else "0"
device = 1 if torch.cuda.is_available() else None

trainer = pl.Trainer(gpus=device,  # or None if CPU is used
                     max_steps=10,  # number of training steps
                     logger=False,
                     benchmark=True,
                     enable_checkpointing=False)

trainer.fit(solver)  # start training

optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.05,
                            optimizer_args={'max_iter': 2, 'history_size': 100})

# LBFGS can not work with varying points!
pde_cond.sampler = pde_cond.sampler.make_static()
solver = tp.solver.Solver(
    train_conditions=[bound_cond, pde_cond], optimizer_setting=optim)

trainer = pl.Trainer(gpus=device,
                     max_steps=30,  # number of training steps
                     logger=False,
                     benchmark=True)

trainer.fit(solver)

plot_sampler = tp.samplers.PlotSampler(
    plot_domain=square, n_points=640, device='cuda')

fig = tp.utils.plot(model, lambda u: u, plot_sampler,
                    plot_type='contour_surface')

inp_points, output, _ = tp.utils.plotting.plot_functions._create_plot_output(
    model, lambda u: u, plot_sampler, device="cuda")

domain_points = tp.utils.plotting.plot_functions._extract_domain_points(inp_points, plot_sampler.domain,
                                                                        len(plot_sampler))

coordinates = output["x"].detach().numpy()
values = output["u"].detach().numpy()

np.savetxt("coordinates.txt", coordinates)
np.savetxt("values.txt", values)
# plt.savefig("test.png")

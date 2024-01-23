"""Defines classes and functions for the post processing of your experiment."""
import matplotlib.pyplot as plt
import numpy as np

from yotse.pre import Experiment


def plot_cost_function(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Plot the cost function."""
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_surface(
        x, y, z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3
    )

    ax.contour(x, y, z, zdir="z", offset=np.min(z), cmap="coolwarm")
    ax.contour(x, y, z, zdir="x", offset=np.min(x), cmap="coolwarm")
    ax.contour(x, y, z, zdir="y", offset=np.max(y), cmap="coolwarm")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


def plot_opt_steps(
    experiment: Experiment, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> None:
    """Plot the optimization steps."""
    # data = collect_all_data()

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_surface(
        x, y, z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3
    )

    # To be fixed
    plt.xlabel(experiment.parameters[0].name)
    plt.ylabel(experiment.parameters[1].name)

    plt.show()

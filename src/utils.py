"""

UTILITY FUNCTIONS FOR DeepFRI
Author: Abijith J. Kamath, Indian Institute of Science

"""

import torch
import math
import numpy as np
import os

import matplotlib.ticker as ticker

from matplotlib import pyplot as plt
from torch.nn import functional as F
from scipy import linalg as splin


def dirichlet(t, bandwidth, duration):
    """Returns point evaluation of the Dirichlet kernel

    Arguments:
        t (np.ndarray): evaluation points
        bandwidth (float): bandwidth of the kernel
        duration (float): time support of the kernel
    Returns:
        np.ndarray: dirichlet kernel
    """
    numerator = np.sin(np.pi * bandwidth * t)
    denominator = bandwidth * duration * np.sin(np.pi * t / duration)

    idx = np.abs(denominator) < 1e-12

    numerator[idx] = np.cos(np.pi * bandwidth * t[idx])
    denominator[idx] = np.cos(np.pi * t[idx] / duration)
    return numerator / denominator


def emoms(t, P):
    P = int(P)

    lambda_0 = 2 * np.pi / (P + 1)
    phi = np.ones(t.shape)
    for i in range(1, P // 2):
        phi += 2 * np.cos(i * lambda_0 * (t - P // 2))

    return phi / (P + 1)


def moms_forward_operator(ak, tk, samples, P, N):
    alphas = 1j * np.pi / (P + 1) * np.arange(0, P + 1)
    V = np.exp(np.outer(alphas, np.arange(N)))

    vec = ak * np.exp(1j * 2 * np.pi * tk)
    s = np.exp(1j * 2 * np.pi * np.outer(np.arange(0, P + 1), tk)) @ vec

    rhs = V @ samples
    cmn = s / rhs

    # diag_mtx[0] = 1
    return np.diag(cmn) @ V


def forward_matrix(time_samples, model_order, period, mode=None):
    """Constructs matrix of forward transform to Fourier series coefficients
    given time samples, period and model order.

    In the special case under critical uniform sampling,
    return the DFT matrix

    :param time_samples: locations of samples
    :param model_order: number of Fourier series coefficients
    :param period: period of the FRI signal

    :returns: forward matrix

    """

    K = model_order
    if mode == None:
        return (
            np.exp(
                1j * 2 * np.pi * np.outer(time_samples, np.arange(-K, K + 1)) / period
            )
            / period
        )

    elif mode == "iftem":
        num_samples = len(time_samples)
        F1 = np.exp(
            1j * 2 * np.pi * np.outer(time_samples[1::], np.arange(-K, K + 1)) / period
        )
        F2 = np.exp(
            1j
            * 2
            * np.pi
            * np.outer(time_samples[0 : num_samples - 1], np.arange(-K, K + 1))
            / period
        )
        F = F1 - F2
        F[:, K] = time_samples[1::] - time_samples[0 : num_samples - 1]

        scale_matrix = period / (1j * 2 * np.pi * np.arange(-K, K + 1))
        scale_matrix[K] = 1
        scale_matrix = np.diag(scale_matrix)

        return np.matmul(F, scale_matrix)

    elif mode == "dftem":
        F = np.exp(
            1j * 2 * np.pi * np.outer(time_samples, np.arange(-K, K + 1)) / period
        )

        scale_matrix = 1j * 2 * np.pi * np.arange(-K, K + 1)
        scale_matrix[K] = 1
        scale_matrix = np.diag(scale_matrix)

        return np.matmul(F, scale_matrix)


def toeplitzification(x, M, P):
    """Toeplitzification of an odd length vector

    Arguments:
        x (np.ndarray): generator
        M (int): dimension
        P (int): dimension

    Returns:
        np.ndarray: lifted toeplitz matrix of x
    """

    N = 2 * M + 1
    index_i = -M + P + np.arange(1, N - P + 1) - 1
    index_j = np.arange(1, P + 2) - 1
    index_ij = index_i[:, None] - index_j[None, :]
    Tp_x = x[index_ij + M]
    return Tp_x


def adj_toeplitzification(x, N, P):
    """Adjoint of Toeplitzification

    Arguments:
        x (np.ndarray): toeplitz matrix
        N (int): dimension
        P (int): dimension

    Returns:
        np.ndarray: adjoint of toeplitz matrix
    """

    offsets = -(np.arange(1, N + 1) - 1 - P)
    out = np.zeros(shape=(N,), dtype=x.dtype)
    for (i, m) in enumerate(offsets):
        out[i] = np.sum(np.diagonal(x, offset=m))

    return out


def toep_gram(P, N):
    """Gram matrix of toeplitzification

    Arguments:
        P (int): dimension
        N (int): dimension

    Returns:
        Diagonal elements of gram matrix

    """

    weights = np.ones(shape=(N,)) * (P + 1)
    weights[:P] = np.arange(1, P + 1)
    weights[N - P :] = np.flip(np.arange(1, P + 1), axis=0)

    return weights


def pinv_toeplitzification(x, N, P):
    """Pseudoinverse of toeplitzification operator

    Arguments:
        x (np.ndarray): toeplitz matrix
        N (int): dimension
        P (int): dimension

    Returns:
        np.ndarray: generator of the toeplitz matrix
    """

    return adj_toeplitzification(x, N, P) / toep_gram(P, N)


def low_rank_approximation(data, rank):
    """Low-rank approxiation of data with rank

    Arguments:
        data (np.ndarray): input matrix
        rank (int): target rank

    Returns:
        np.ndarray: low rank approximation of data
    """
    u, s, vh = splin.svd(data, full_matrices=False, check_finite=False)
    return (u[:, :rank] * s[None, :rank]) @ vh[:rank, :]


def get_tau(G, P, select=None):
    """Compute PGD parameter that guarantees convergence

    Arguments:
        G (np.ndarray): forward matrix
        P (int): model order

    Returns:
        float: largest PGD parameter that guarantees convergence
    """
    # eig_vals, _ = np.linalg.eig(np.conj(G).T @ G)
    eig_vals, _ = np.linalg.eig(np.linalg.pinv(G) @ G)
    beta = 2 * np.max(eig_vals)

    lower_lim = np.abs((1 - 1 / np.sqrt(P + 1)) / beta)
    upper_lim = np.abs((1 + 1 / np.sqrt(P + 1)) / beta)

    if select == "low":
        return lower_lim + 1e-4
    elif select == "high":
        return upper_lim - 1e-4
    elif select == "average":
        return 0.5 * (lower_lim + upper_lim)
    else:
        return lower_lim + (upper_lim - lower_lim) * np.random.rand()


def normalised_mean_squared_error(a, b):
    """Computes mean-squared-error between a and b

    Arguments:
        a (np.ndarray): input vector
        b (np.ndarray): input vector

    Returns:
        nmse between a and b
    """
    return np.abs(np.mean((a - b) ** 2)) / np.abs(np.mean((a) ** 2))


def gen_noise(sigma, shape):
    return sigma * torch.normal(std=1, mean=0, size=shape)


def gen_noise_psnr(y, a_kmax, psnr_vec):

    bsz, N = y.shape[0], y.shape[-1]

    tmp = 10 ** (-psnr_vec / 20)
    sigmas = tmp * a_kmax
    noise = torch.randn(y.size(), dtype=torch.complex128)
    noiseSqrtPower = torch.linalg.vector_norm(noise, 2, dim=-1) / math.sqrt(N)
    noise = (sigmas / noiseSqrtPower)[:, None] * noise
    noise = noise.to(torch.complex128)

    return noise


def compute_sigma_psnr(a_kmax, psnr):
    sigma_vec = (10 ** (-psnr / 20)) * a_kmax
    return (sigma_vec)[:, None]


def load_data(params, mode, seperations=None, ultrasound=False, use_emoms=False):
    parent_dir = os.path.dirname(os.getcwd())
    data_folder = f"data/K{params.K}/{mode}"
    data_path = os.path.join(parent_dir, data_folder)

    if ultrasound:
        filename = f"ultrasound_synthetic"
    elif params.K == 2:
        filename = (
            f"{mode}_K{params.K}_gamma{params.gamma}_{seperations[params.index]:0.5f}"
        )
    else:
        filename = f"{mode}_K{params.K}_gamma{params.gamma}"

    if use_emoms:
        filename = "EMOMS_" + filename

    data = torch.load(os.path.join(data_path, filename + ".pth"))
    a = np.load(os.path.join(data_path, "amplitudes_" + filename + ".npy"))
    t = np.load(os.path.join(data_path, "locations_" + filename + ".npy"))
    return data, a, t, parent_dir


############################## PLOTTING TOOLS #####################################


def plot_diracs(
    tk,
    ak,
    ax=None,
    plot_colour="blue",
    alpha=1,
    line_width=2,
    marker_style="o",
    marker_size=4,
    line_style="-",
    legend_show=True,
    legend_loc="lower left",
    legend_label=None,
    ncols=2,
    title_text=None,
    xaxis_label=None,
    yaxis_label=None,
    xlimits=[0, 1],
    ylimits=[-1, 1],
    show=False,
    save=None,
):
    """Plots Diracs at tk, ak"""
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

    markerline, stemlines, baseline = plt.stem(
        tk, ak, label=legend_label, linefmt=line_style
    )
    plt.setp(stemlines, linewidth=line_width, color=plot_colour, alpha=alpha)
    plt.setp(
        markerline,
        marker=marker_style,
        linewidth=line_width,
        alpha=alpha,
        markersize=marker_size,
        markerfacecolor=plot_colour,
        mec=plot_colour,
    )
    plt.setp(baseline, linewidth=0)

    if legend_label and legend_show:
        plt.legend(
            ncol=ncols, loc=legend_loc, frameon=True, framealpha=0.8, facecolor="white"
        )

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))

    if save:
        plt.savefig(save + ".pdf", format="pdf")

    if show:
        plt.show()

    return


def plot_signal(
    x,
    y,
    ax=None,
    plot_colour="blue",
    alpha=1,
    xaxis_label=None,
    yaxis_label=None,
    title_text=None,
    legend_label=None,
    legend_show=True,
    legend_loc="lower left",
    n_col=2,
    line_style="-",
    line_width=None,
    xlimits=[-2, 2],
    ylimits=[-2, 2],
    axis_formatter="%0.1f",
    show=False,
    save=None,
):
    """
    Plots signal with abscissa in x and ordinates in y

    """
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

    plt.plot(
        x,
        y,
        linestyle=line_style,
        linewidth=line_width,
        color=plot_colour,
        label=legend_label,
        zorder=0,
        alpha=alpha,
    )
    if legend_label and legend_show:
        plt.legend(
            ncol=n_col, loc=legend_loc, frameon=True, framealpha=0.8, facecolor="white"
        )

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if axis_formatter:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(axis_formatter))

    if save:
        plt.savefig(save + ".pdf", format="pdf")

    if show:
        plt.show()

    return


def plot_points_on_xaxis(
    points,
    ax=None,
    line_width=None,
    point_colour="black",
    alpha=1,
    legend_label=None,
    legend_show=True,
    legend_loc="lower left",
    title_text=None,
    show=False,
    xlimits=[0, 1],
    ylimits=[-1, 1],
    save=None,
):
    """
    Plots points on x-axis

    """
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

    zerovec = np.zeros(len(points))
    plt.scatter(
        points,
        zerovec,
        zorder=10,
        marker="o",
        alpha=alpha,
        color=point_colour,
        linewidth=line_width,
        label=legend_label,
    )
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)

    if legend_label and legend_show:
        plt.legend(
            ncol=2, loc=legend_loc, frameon=True, framealpha=0.8, facecolor="white"
        )

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.title(title_text)

    if save:
        plt.savefig(save + ".pdf", format="pdf")

    if show:
        plt.show()

    return


def plot_hline(
    level=0,
    ax=None,
    line_colour="black",
    line_style="-",
    alpha=1,
    line_width=0.5,
    annotation=None,
    pos=(1, 1),
):
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

    plt.axhline(
        level,
        color=line_colour,
        linestyle=line_style,
        linewidth=line_width,
        alpha=alpha,
    )
    if annotation:
        plt.annotate(annotation, xy=pos, color=line_colour)


def plot_dirac_clouds(
    tk,
    ak,
    ax=None,
    plot_colour="blue",
    alpha=1,
    line_width=2,
    marker_style="*",
    marker_size=4,
    legend_show=True,
    legend_loc="lower left",
    legend_label=None,
    title_text=None,
    xaxis_label=None,
    yaxis_label=None,
    xlimits=[0, 1],
    ylimits=[-1, 1],
    show=False,
    save=None,
):
    """Scatter plots of Diracs at tk, ak"""

    plt.scatter(
        tk,
        ak,
        color=plot_colour,
        s=marker_size,
        alpha=alpha,
        marker=marker_style,
        edgecolors=plot_colour,
        label=legend_label,
    )

    if legend_label and legend_show:
        plt.legend(
            ncol=2, loc=legend_loc, frameon=True, framealpha=0.8, facecolor="white"
        )

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))

    if save:
        plt.savefig(save + ".pdf", format="pdf")

    if show:
        plt.show()

    return


def plot_mcerrors(
    x,
    y,
    ax=None,
    plot_colour="blue",
    line_width=2,
    marker_style="o",
    marker_size=4,
    line_style="-",
    legend_label=None,
    legend_loc="lower left",
    legend_show=True,
    title_text=None,
    dev_alpha=0.5,
    xaxis_label=None,
    yaxis_label=None,
    xlimits=[-30, 30],
    ylimits=[1e-4, 1e2],
    show=False,
    save=None,
):
    """Plot x,y on loglog"""

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

    # means = np.mean(y, axis=1)
    # devs = np.std(y, axis=1)
    median = np.median(y, axis=1)
    q1 = np.quantile(y, 0.25, axis=1)
    q3 = np.quantile(y, 0.75, axis=1)
    plt.loglog(
        x,
        median,
        linestyle=line_style,
        linewidth=line_width,
        color=plot_colour,
        marker=marker_style,
        markersize=marker_size,
        label=legend_label,
    )
    plt.fill_between(x, q1, q3, color=plot_colour, linewidth=0, alpha=dev_alpha)

    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor="white")

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + ".pdf", format="pdf")

    if show:
        plt.show()

    return


def plot_position_estimates(
    tk,
    tk_estimate,
    jitter_idx,
    ax=None,
    plot_colour="blue",
    marker_style="o",
    marker_size=4,
    legend_label=None,
    legend_loc="lower right",
    legend_show=True,
    title_text=None,
    xaxis_label=None,
    yaxis_label=None,
    xlimits=[0, 1],
    ylimits=[0, 1],
    plot_line=False,
    show=True,
    save=None,
):
    """2D scatter plot for true position vs estimated position"""

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()

    estimate = np.mean(tk_estimate[jitter_idx, :, :], axis=0)
    devs = np.std(tk_estimate[jitter_idx, :, :], axis=0)
    # plt.scatter(estimate, tk, marker=marker_style, color=plot_colour,
    #     s=marker_size, linewidth=line_width, label=legend_label)
    plt.errorbar(
        estimate,
        tk,
        xerr=devs,
        fmt=marker_style,
        color=plot_colour,
        ms=marker_size,
        capsize=10,
        alpha=0.8,
        label=legend_label,
    )

    if plot_line:
        t = np.linspace(xlimits[0], xlimits[1])
        plt.plot(t, t, color="black", linestyle="-", zorder=0)

    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor="white")

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + ".pdf", format="pdf")

    if show:
        plt.show()

    return

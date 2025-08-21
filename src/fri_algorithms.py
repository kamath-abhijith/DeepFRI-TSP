"""

FRI ALGORITHMS IMPLEMENTATION

ALSO SEE:
    [1] https://github.com/matthieumeo/pyoneer
    [2] https://github.com/hanjiepan/FRI_pkg

"""

import numpy as np
import scipy.optimize as scop

from scipy import linalg
from astropy import units as uts
from astropy import coordinates as crd
from src.utils import *


def cadzow_ls(x, M, P, rank, rho=None, num_iter=10):
    """Cadzow denoising of x with rank

    Arguments
        x (np.ndarray): data vector
        M (int): dimension
        P (int): dimension
        rank (int): rank threshold
        rho (float): projection radius
        num_iter (int): number of iterations

    Returns:
        np.ndarray: denoised data
    """
    N = 2 * M + 1
    if rho:
        for _ in range(num_iter):
            # x = proj_l2_ball(x, rho)
            x = toeplitzification(x, M, P)
            x = low_rank_approximation(x, rank)
            x = pinv_toeplitzification(x, N, P)
    else:
        for _ in range(num_iter):
            x = toeplitzification(x, M, P)
            x = low_rank_approximation(x, rank)
            x = pinv_toeplitzification(x, N, P)

    return x


def cadzow_ls_torch(z, M, P, rank, num_iter=10):
    """Cadzow denoising of x with rank (pytorch implementation)

    Arguments
        x (np.ndarray): data vector
        M (int): dimension
        P (int): dimension
        rank (int): rank threshold
        rho (float): projection radius
        num_iter (int): number of iterations

    Returns:
        np.ndarray: denoised data
    """
    z0 = z[0].cpu().detach().numpy()
    tmp = cadzow_ls(z0, M, P, rank, rho=None, num_iter=num_iter)
    out = torch.tensor(tmp, dtype=torch.complex128, device=z.device)
    out = out[None, :]

    for itr in range(1, z.shape[0]):
        z_batch = z[itr].cpu().detach().numpy()
        tmp = cadzow_ls(z_batch, M, P, rank, rho=None, num_iter=num_iter)
        tensor = torch.tensor(tmp, dtype=torch.complex128, device=z.device)
        tensor = tensor[None, :]
        out = torch.cat((out, tensor))

    return out


def cpgd(G, y, tau, P, rank, rho, init, tol=1e-12, num_iter=50):
    """Cadzow Plug-n-Play Gradient Descent algorithm

        G (np.ndarray): forward matrix
        y (np.ndarray): measurements
        tau (float): parameter for PGD
        P (int): model order
        rank (int): rank threshold
        rho (float): radius of threshold
        init (np.ndarray): initialisation

    Returns:
        np.ndarray: fourier series coefficients
    """

    _, N = G.shape
    M = int(N // 2)

    x = init
    for itr in range(num_iter):
        der = 2 * np.conj(G).T @ (G @ x - y)
        z = x - tau * der
        x = cadzow_ls(z, M, P, rank, rho)
    return x


def cpdm(G, y, tau, P, rank, rho, init, tol=1e-12, num_iter=50, momentum=False):
    """Cadzow Plug-n-Play Gradient Descent algorithm

        G (np.ndarray): forward matrix
        y (np.ndarray): measurements
        tau (float): parameter for PGD
        P (int): model order
        rank (int): rank threshold
        rho (float): radius of threshold
        init (np.ndarray): initialisation

    Returns:
        np.ndarray: fourier series coefficients
    """

    _, N = G.shape
    M = int(N // 2)
    target = np.linalg.pinv(G) @ y

    x = init
    z = x
    t = 1
    for itr in range(num_iter):
        t_old = t
        x_old = x

        x = cadzow_ls(z - tau * (z - target), M, P, rank, rho)

        if not momentum:
            z = x
        else:
            t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
            z = x + (t_old - 1) / t * (x - x_old)

        if np.linalg.norm(x_old - x) / np.linalg.norm(x) < tol:
            break

    return x


def fcpgd_emoms(C, y, tau, P, rank, rho, init, tol=1e-12, num_iter=50, momentum=False):
    """Cadzow Plug-n-Play Gradient Descent algorithm

        G (np.ndarray): forward matrix
        y (np.ndarray): measurements
        tau (float): parameter for PGD
        P (int): model order
        rank (int): rank threshold
        rho (float): radius of threshold
        init (np.ndarray): initialisation

    Returns:
        np.ndarray: fourier series coefficients
    """

    _, N = C.shape
    M = int(N // 2)
    target = C @ y

    x = init
    z = x
    t = 1
    for itr in range(num_iter):
        t_old = t
        x_old = x

        x = cadzow_ls(z - tau * (z - target), M, P, rank, rho)

        if not momentum:
            z = x
        else:
            t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
            z = x + (t_old - 1) / t * (x - x_old)

        if np.linalg.norm(x_old - x) / np.linalg.norm(x) < tol:
            break

    return x


def prony_tls(swce, model_order):
    """High resolution spectral estimation (HRSE) using Annihilating Filter

    Arguments:
        swce (np.ndarray): input linear combination of sinusoids
        model_order (int): number of sinusoids

    Returns:
        np.ndarray: annihilating filter
    """

    N = len(swce)
    M = int(N // 2)

    index_i = -M + model_order + np.arange(1, N - model_order + 1) - 1
    index_j = np.arange(1, model_order + 2) - 1
    index_ij = index_i[:, None] - index_j[None, :]
    conv_mtx = swce[index_ij + M]

    _, _, vh = linalg.svd(conv_mtx, check_finite=False, full_matrices=False)
    annihilating_filter = vh[-1, :].conj()
    annihilating_filter = annihilating_filter.reshape(-1)

    return annihilating_filter


def get_shifts(annihilating_filter, support, real_data=False, use_emoms=False):
    """Get shifts from the annihilating filter

    Arguments:
        annihilating_filter (np.ndarray): coefficients of the annihilating filter
        support (int): support of the signal in the time domain

    Returns:
        np.ndarray: dirac locations
    """

    if use_emoms:
        roots = np.roots(-annihilating_filter)
        locations = crd.Angle(np.angle(roots) * uts.rad)
        locations = locations.wrap_at(2 * np.pi * uts.rad)
        locations = locations.value.reshape(-1) / (2 * np.pi)
        return np.sort(locations)

    if real_data:
        roots = np.roots(annihilating_filter)
        locations = crd.Angle(-np.angle(roots) * uts.rad)

    else:
        roots = np.roots(np.flip(annihilating_filter, axis=0).reshape(-1))
        locations = crd.Angle(np.angle(roots) * uts.rad)

    locations = locations.wrap_at(2 * np.pi * uts.rad)
    if real_data:
        return np.sort(locations.value.reshape(-1) / support)
    else:
        return np.sort(support * locations.value.reshape(-1) / (2 * np.pi))


def match_to_ground_truth(true_locations, estimated_locations, period):
    """Match estimated sources to ground truth with a bipartite graph matching algorithm.
        true_locations: np.ndarray[K,],true  dirac locations.
        estimated_locations: np.ndarray[K,], estimated dirac locations.

    Returns:
        np.ndarray: estimated locations is reordered to match true locations
        float: Average cost of matching (positionning error).
    """
    true_locations = true_locations.reshape(-1)
    distance = np.abs(true_locations[:, None] - estimated_locations[None, :])
    cost = np.fmin(distance, period - distance)
    row_ind, col_ind = scop.linear_sum_assignment(cost)
    return estimated_locations[col_ind], cost[row_ind, col_ind].mean()


def get_weights(swce, locations, tau, tikh=None, landweber=None):
    """Returns the weights of the FRI signal given the locations

    Arguments:
        swce: sequence in sum-of-weighted-complex-exponentials form
        locations: estimated locations of the Dirac impulses
        tau: period of the signal

    Returns:
        np.ndarray: weights of the FRI signal
    """

    M = int(len(swce) // 2)
    K = len(locations)
    swce_matrix = np.exp(
        -1j * 2 * np.pi * np.outer(np.arange(-M, M + 1), locations) / tau
    )

    if tikh is not None:
        A = np.conj(swce_matrix).T @ swce_matrix + tikh * np.eye(K)
        b = np.conj(swce_matrix).T @ swce
        return (2 * M + 1) * np.linalg.solve(A, b)
    elif landweber is not None:
        pass
    else:
        return (2 * M + 1) * np.linalg.pinv(swce_matrix) @ swce


def Tmtx(data, K):
    """Construct convolution matrix for a filter specified by 'data'"""
    return linalg.toeplitz(data[K::], data[K::-1])


def Rmtx(data, K, seq_len):
    """A dual convolution matrix of Tmtx. Use the commutativness of a convolution:
    a * b = b * c
    Here seq_len is the INPUT sequence length
    """
    col = np.concatenate(([data[-1]], np.zeros(seq_len - K - 1)))
    row = np.concatenate((data[::-1], np.zeros(seq_len - K - 1)))
    return linalg.toeplitz(col, row)


def gen_fri(G, a, K, noise_level=0, max_ini=100, stop_cri="mse"):
    compute_mse = stop_cri == "mse"
    M = G.shape[1]
    GtG = np.dot(G.conj().T, G)
    Gt_a = np.dot(G.conj().T, a)

    max_iter = 50
    min_error = float("inf")
    # beta = linalg.solve(GtG, Gt_a)
    beta = linalg.lstsq(G, a)[0]

    Tbeta = Tmtx(beta, K)
    rhs = np.concatenate((np.zeros(2 * M + 1), [1.0]))
    rhs_bl = np.concatenate((Gt_a, np.zeros(M - K)))

    for ini in range(max_ini):
        c = np.random.randn(K + 1) + 1j * np.random.randn(K + 1)
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = Rmtx(c, K, M)

        # first row of mtx_loop
        mtx_loop_first_row = np.hstack(
            (
                np.zeros((K + 1, K + 1)),
                Tbeta.conj().T,
                np.zeros((K + 1, M)),
                c0[:, np.newaxis],
            )
        )
        # last row of mtx_loop
        mtx_loop_last_row = np.hstack(
            (c0[np.newaxis].conj(), np.zeros((1, 2 * M - K + 1)))
        )

        for loop in range(max_iter):
            mtx_loop = np.vstack(
                (
                    mtx_loop_first_row,
                    np.hstack(
                        (Tbeta, np.zeros((M - K, M - K)), -R_loop, np.zeros((M - K, 1)))
                    ),
                    np.hstack(
                        (np.zeros((M, K + 1)), -R_loop.conj().T, GtG, np.zeros((M, 1)))
                    ),
                    mtx_loop_last_row,
                )
            )

            # matrix should be Hermitian symmetric
            mtx_loop += mtx_loop.conj().T
            mtx_loop *= 0.5
            # mtx_loop = (mtx_loop + mtx_loop.conj().T) / 2.

            c = linalg.solve(mtx_loop, rhs)[: K + 1]

            R_loop = Rmtx(c, K, M)

            mtx_brecon = np.vstack(
                (
                    np.hstack((GtG, R_loop.conj().T)),
                    np.hstack((R_loop, np.zeros((M - K, M - K)))),
                )
            )

            # matrix should be Hermitian symmetric
            mtx_brecon += mtx_brecon.conj().T
            mtx_brecon *= 0.5
            # mtx_brecon = (mtx_brecon + mtx_brecon.conj().T) / 2.

            b_recon = linalg.solve(mtx_brecon, rhs_bl)[:M]

            error_seq[loop] = linalg.norm(a - np.dot(G, b_recon))
            if error_seq[loop] < min_error:
                min_error = error_seq[loop]
                b_opt = b_recon
                c_opt = c
            if min_error < noise_level and compute_mse:
                break
        if min_error < noise_level and compute_mse:
            break

    return b_opt, min_error, c_opt, ini


def get_tks(fourier_coeffs, tk_ground, M, T, annihilating_filter=None, use_emoms=False):
    if annihilating_filter is None:
        annihilating_filter = prony_tls(fourier_coeffs, M)
    tk_estimate = get_shifts(annihilating_filter, T, use_emoms=use_emoms)
    t_k, _ = match_to_ground_truth(tk_ground, tk_estimate, period=T)
    return t_k

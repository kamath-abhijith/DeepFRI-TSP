"""

TESTING MODULES FOR DeepFRI
Author: Sharan Patil, Indian Institute of Science

Contains parallelisable functions to test FRI reconstruction algorithms

Each function returns the normalised mean squared error between the estimated
locations and the ground truth locations of the diracs.

"""

import numpy as np
from src.utils import *
from src.fri_algorithms import *

def test_fri_algorithm(i, Y_noisy, t, G, M, K, T, algorithm, use_emoms=False):
    """ """
    Y = Y_noisy[i, :].numpy()
    tk_ground = t[i, :]
    annihilating_filter = None

    if algorithm == "cadzow":

        noisy_fourier_coeffs = np.squeeze(np.linalg.pinv(G) @ Y)
        # print(noisy_fourier_coeffs.dtype)
        fourier_coeffs = cadzow_ls(
            noisy_fourier_coeffs, P=M, M=M, rank=K, rho=None, num_iter=10
        )
    elif algorithm == "cpgd":

        init = np.zeros(2 * M + 1)
        tau = get_tau(G, P=M)
        # print(tau)

        fourier_coeffs = cpgd(
            G,
            np.squeeze(Y),
            P=M,
            tau=tau,
            rank=K,
            rho=np.linalg.norm(np.squeeze(Y),2),
            init=init,
            tol=1e-12,
            num_iter=50,
        )
    elif algorithm == "cpdm":
        P = M
        init = np.zeros(2 * M + 1)
        # tau = get_tau(G, P=M)
        tau = 1 - (1 / (2 * np.sqrt(P + 1)))
        # tau = (tau + 1) / 2
        # print(tau)

        fourier_coeffs = cpdm(
            G,
            np.squeeze(Y),
            P=M,
            tau=tau,
            rank=K,
            rho=None,
            init=init,
            tol=1e-12,
            num_iter=50,
        )
    elif algorithm == "genfri":
        [fourier_coeffs, _, annihilating_filter, _] = gen_fri(G, np.squeeze(Y), K)

    elif use_emoms:
        P = M
        init = np.zeros(2 * M + 1)
        # tau = get_tau(G, P=M)
        tau = 1 - (1 / (2 * np.sqrt(P + 1)))
        # tau = (tau + 1) / 2
        # print(tau)

        fourier_coeffs = fcpgd_emoms(
            G,
            np.squeeze(Y),
            P=M,
            tau=tau,
            rank=K,
            rho=None,
            init=init,
            tol=1e-12,
            num_iter=150,
        )

    else:
        raise Exception("Invalid Algorithm Selected")
    tk_estimate = get_tks(
        fourier_coeffs,
        tk_ground,
        M,
        T,
        annihilating_filter=annihilating_filter,
        use_emoms=use_emoms,
    )

    return normalised_mean_squared_error(tk_ground, tk_estimate)


def test_deepfri(i, fourier_coeffs, t, M, T, use_emoms=False):
    tk_ground = t[i, :]
    tk_estimate = get_tks(fourier_coeffs[i, :], tk_ground, M, T, use_emoms=use_emoms)

    # if use_emoms:
    #     return tk_ground[0] - tk_estimate[0]
    # else:
    #     return normalised_mean_squared_error(tk_ground, tk_estimate)
    return normalised_mean_squared_error(tk_ground, tk_estimate)

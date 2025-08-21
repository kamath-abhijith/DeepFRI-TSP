"""

HELPER FUNCTIONS FOR DATASET GENERATION FOR TRAINING DeepFRI
Author: Sharan Patil, Indian Institute of Science

"""

import torch
import numpy as np
from src.utils import dirichlet, emoms

def generate_samples(a_k, t_k, N, T, sampling_times):
    """Generate samples of a dirichlet filtered dirac comb

    Arguments:
        a_k (np.ndarray): amplitudes of the diracs
        t_k (np.ndarray): locations of the diracs
        N (int): number of samples
        T (float): period of the dirac comb
        sampling_times (np.ndarray): time stamps where the signal is sampled

    Returns:
        torch.Tensor: samples of the dirichlet filtered dirac comb
    """
    bandwidth = N / T
    tl_grid, tk_grid = np.meshgrid(t_k, sampling_times)
    samples = np.inner(a_k, dirichlet(tk_grid - tl_grid, bandwidth, T))
    return torch.tensor(samples, dtype=torch.complex128)


def generate_fourier_coefficients(a_k, t_k, M, T):
    """Generate samples of a dirichlet filtered dirac comb

    Arguments:
        a_k (np.ndarray): amplitudes of the diracs
        t_k (np.ndarray): locations of the diracs
        M (int): consecutive fourier coefficients
        T (float): period of the dirac comb
        sampling_times (np.ndarray): time stamps where the signal is sampled

    Returns:
        torch.Tensor: 2*M + 1 consecutive fourier coefficients
    """
    m_arr = np.arange(-M, M + 1)
    mat = np.outer(m_arr, t_k)
    g = np.exp(-1j * 2 * np.pi * mat / T)
    x_hat = g @ a_k
    return torch.tensor(x_hat, dtype=torch.complex128)

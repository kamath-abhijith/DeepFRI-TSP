"""

DATASET GENERATION FOR DeepFRI
Author: Sharan Patil, Indian Institute of Science

"""

import torch
import os
import numpy as np

from src.sampling import *

def generate_K2(relative_minimal_distance, mode):
    """Generates two diracs

    The first dirac is fixed at t = 0.1 seconds. The location of the second
    dirac is randomly generated in the case of training, and is fixed in place
    in the case of testing.

    Arguments:
        relative_minimal_distance (float): minimum seperation (in seconds) between the diracs
        mode (str): specify to generate train or test data

    Returns:
        np.ndarray: two diracs
    """
    if mode == "test":
        dist = relative_minimal_distance
    elif mode == "train":
        dist = np.random.rand() * (2 * relative_minimal_distance)
    else:
        raise Exception("Invalid Mode Selected")

    return np.array([0.1, 0.1 + dist])


def generate_data(
    num_data,
    K,
    N,
    M,
    T,
    sampling_times,
    relative_minimal_distance=0.01,
    mode=None,
    t_start=0.0,
    t_end=1.0,
    use_emoms=False,
):
    """Generates a dataset for training/testing

    Each datapoint in the dataset corresponds to a randomly generated signal.
    Each datapoint containing samples, fourier coefficients, amplitudes of the
    diracs and locations of the diracs.

    Arguments:
        num_data (int): number of datapoints to be generated
        K (int): number of diracs in each signal generated
        N (int): number of samples
        M (int): dimension
        T (float): time period of the signal
        sampling_times (np.ndarray): time stamps where the signal is sampled
        relative_minimal_distance (float): minimum seperation between the diracs
        mode (str): specify to generate train or test data
        t_start (float): start of the time period
        t_end (float): end of the time period

    Returns:
        tensordataset: dataset containing (sampled_signal, fourier_coefficient) pairs
        np.ndarray: amplitudes of the diracs in each signal
        np.ndarray: locations of the diracs in each signal


    """
    samples = torch.zeros((num_data, N), dtype=torch.complex128)
    coefficients = torch.zeros((num_data, 2 * M + 1), dtype=torch.complex128)
    grid = np.arange(t_start, t_end, relative_minimal_distance * (t_end - t_start))

    a = np.zeros((num_data, K))
    t = np.zeros((num_data, K))

    for i in range(num_data):

        if use_emoms:
            # a_k = np.random.uniform(-10, 10, K)
            # a_k = np.random.randn(K)
            if mode == "train":
                a_k = np.random.normal(0, np.sqrt(1), K)
            elif mode == "test":
                a_k = np.random.uniform(-0.5, 10, K)
        else:
            a_k = np.random.randn(K)

        if K == 2:
            t_k = generate_K2(relative_minimal_distance, mode)
        else:
            t_k = np.sort(
                grid[
                    np.random.permutation(np.arange(grid.size))[:K].astype(int)
                ].reshape(-1)
            )

        samples[i][:] = generate_samples(a_k, t_k, N, T, sampling_times)
        coefficients[i][:] = generate_fourier_coefficients(a_k, t_k, M, T)

        a[i, :] = a_k
        t[i, :] = t_k

    return torch.utils.data.TensorDataset(samples, coefficients), (a, t)


def generate_and_save_data(
    num_data,
    K,
    N,
    M,
    T,
    sampling_times,
    save_name,
    relative_minimal_distance=0.01,
    mode=None,
    use_emoms=False,
):
    """Generates and saves a dataset

    Arguments:
        num_data (int): number of datapoints to be generated
        K (int): number of diracs in each signal generated
        N (int): number of samples
        M (int): dimension
        T (float): time period of the signal
        sampling_times (np.ndarray): time stamps where the signal is sampled
        save_name (str): name of the dataset to be saved
        relative_minimal_distance (float): minimum seperation between the diracs
        mode (str): specify to generate train or test data
        t_start (float): start of the time period
        t_end (float): end of the time period

    Returns:
        None
    """
    dataset, (amplitudes, locations) = generate_data(
        num_data=num_data,
        K=K,
        N=N,
        M=M,
        T=T,
        relative_minimal_distance=relative_minimal_distance,
        sampling_times=sampling_times,
        mode=mode,
        use_emoms=use_emoms,
    )

    data_dir = f"data/K{K}/{mode}"
    parent_dir = os.path.dirname(os.getcwd())
    save_dir = os.path.join(parent_dir, data_dir)

    if use_emoms:
        save_name = "EMOMS_" + save_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(dataset, os.path.join(save_dir, save_name + ".pth"))
    np.save(
        os.path.join(save_dir, "locations_" + save_name + ".npy"),
        locations,
    )
    np.save(
        os.path.join(save_dir, "amplitudes_" + save_name + ".npy"),
        amplitudes,
    )

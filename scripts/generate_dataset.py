"""

GENERATE DATASETS FOR TRAINING DeepFRI
Author: Sharan Patil, Indian Institute of Science

"""

import argparse
import numpy as np

from src.dataset import *

parser = argparse.ArgumentParser("Simulation Data")

parser.add_argument("--N", type=int, default=73, help="Number of samples")
parser.add_argument(
    "--K", type=int, default=9, help="Number of teeth in the dirac comb"
)
parser.add_argument("--gamma", type=int, default=2, help="Oversampling Parameter")
parser.add_argument("--T", type=float, default=1.0, help="Time Period [0, T]")
parser.add_argument(
    "--num_data",
    type=int,
    default=100000,
    help="Number of datapoints in the dataset",
)
parser.add_argument("--mode", type=str, default="train", help="train/test")
parser.add_argument(
    "--ultrasound", type=bool, default=False, help="generate ultrasound data"
)
parser.add_argument(
    "--use_emoms", type=bool, default=False, help="generate data using the emoms kernel"
)

params = parser.parse_args()

params.M = params.K * params.gamma
params.sampling_times = np.arange(0, params.N) * params.T / params.N

if params.ultrasound:
    params.K = 2
    params.mode = "train"
    filename = f"ultrasound_synthetic"
    generate_and_save_data(
        num_data=params.num_data,
        K=params.K,
        N=params.N,
        M=params.M,
        T=params.T,
        sampling_times=params.sampling_times,
        save_name=filename,
        relative_minimal_distance=1e-7,
        mode=params.mode,
        use_emoms=params.use_emoms,
    )

elif params.K != 2:
    filename = f"{params.mode}_K{params.K}_gamma{params.gamma}"

    generate_and_save_data(
        num_data=params.num_data,
        K=params.K,
        N=params.N,
        M=params.M,
        T=params.T,
        sampling_times=params.sampling_times,
        save_name=filename,
        relative_minimal_distance=0.01,
        mode=params.mode,
        use_emoms=params.use_emoms,
    )

elif params.K == 2:
    exponents = np.linspace(-3, -1, num=9)
    seperations = 10 ** (exponents)

    for seperation in seperations:

        filename = f"{params.mode}_K{params.K}_gamma{params.gamma}_{seperation:0.5f}"

        generate_and_save_data(
            num_data=params.num_data,
            K=params.K,
            N=params.N,
            M=params.M,
            T=params.T,
            sampling_times=params.sampling_times,
            save_name=filename,
            relative_minimal_distance=seperation,
            mode=params.mode,
            use_emoms=params.use_emoms,
        )

"""

F-CPGD AND BENCHMARKS TESTING CODE
Author: Sharan Patil, Indian Institute of Science

"""

import sys

from src.utils import *
from src.testing_module import *
from src.lib import *
from src.dataset import *
from scipy.io import loadmat

from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import argparse
import time
import torch

seperations = None
psnrs = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

parser = argparse.ArgumentParser("Training DeepFRI")
parser.add_argument("--N", type=int, default=81, help="Number of samples")
parser.add_argument(
    "--K", type=int, default=10, help="Number of teeth in the dirac comb"
)
parser.add_argument("--gamma", type=int, default=2, help="Oversampling Parameter")
parser.add_argument("--T", type=float, default=1.0, help="Time Period [0, T]")
parser.add_argument(
    "--algorithm", type=str, default="cadzow", help="cadzow/cpgd/cpdm/genfri"
)
parser.add_argument(
    "--use_emoms", type=bool, default=False, help="If the sampling kernel is EMOMS"
)
parser.add_argument("--index", type=int, default=0, help="index for seperation array")
params = parser.parse_args()
params.M = params.K * params.gamma
params.P = params.M
params.sampling_times = np.arange(0, params.N) * params.T / params.N
if params.K == 2:
    exponents = np.linspace(-3, -1, num=9)
    seperations = 10**exponents

test_data, a, t, parent_dir = load_data(
    params, mode="test", seperations=seperations, use_emoms=params.use_emoms
)
a_kmax = np.max(np.abs(a), axis=1)

Y_data = test_data[:][0]
x_data = test_data[:][1] 
a_data = torch.Tensor(a_kmax)
dset = torch.utils.data.TensorDataset(Y_data, x_data, a_data)

Y = dset[:][0]
ak_max = dset[:][2]

Y = Y[:1000][:]
ak_max = ak_max[:1000]

errors = []

## UNIFORM SAMPLING
# t_continuous = np.arange(0, params.N, dtype=float) * params.T / params.N

## IRREGULAR SAMPLING
np.random.seed(20)
relative_minimal_distance = 0.04
grid = np.arange(0, 1, relative_minimal_distance)
t_continuous = np.sort(grid[np.random.permutation(np.arange(grid.size))[:params.N].astype(int)].reshape(-1))

forward_mtx = np.exp(
    1j
    * 2
    * np.pi
    * np.outer(t_continuous, np.arange(-params.M, params.M + 1))
    / params.T
)

print(f"####################### Testing {params.algorithm} {params.index} ####################### ")
for i, psnr in enumerate(psnrs):
    print(f"PSNR={psnr}")

    noise = gen_noise_psnr(Y, ak_max, psnr)
    Y_noisy = Y + noise

    results = Parallel(n_jobs=-1)(
        delayed(test_fri_algorithm)(
            i,
            Y_noisy,
            t,
            forward_mtx,
            params.M,
            params.K,
            params.T,
            params.algorithm,
            use_emoms=params.use_emoms,
        )
        for i in tqdm(range(1000))
    )

    results = np.array(results)

    errors.append(results)


errors = np.array(errors)

results_dir = f"results/K{params.K}/plots/{params.algorithm}"
save_dir = os.path.join(parent_dir, results_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if params.K == 2:
    filename = f"{params.algorithm}_N{params.N}_K{params.K}_gamma{params.gamma}_{seperations[params.index]:0.5f}.npy"
    if params.use_emoms:
        filename = "EMOMS_" + filename
else:
    filename = f"{params.algorithm}_N{params.N}_K{params.K}_gamma{params.gamma}.npy"

np.save(os.path.join(save_dir, filename), errors)

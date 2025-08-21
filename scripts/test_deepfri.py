"""

DeepFRI TESTING CODE
Author: Sharan Patil, Indian Institute of Science

"""


from src.utils import *
from src.testing_module import *
from src.lib import *
from src.dataset import *

from tqdm import tqdm
from scipy.io import loadmat
from joblib import Parallel, delayed

import numpy as np
import argparse
import time
import torch

seperations = None
psnrs = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

parser = argparse.ArgumentParser("Testing DeepFRI")

parser.add_argument("--N", type=int, default=81, help="Number of samples")
parser.add_argument(
    "--K", type=int, default=10, help="Number of teeth in the dirac comb"
)
parser.add_argument("--gamma", type=int, default=1, help="Oversampling Parameter")
parser.add_argument("--T", type=float, default=1.0, help="Time Period [0, T]")
parser.add_argument(
    "--num_unfoldings", type=int, default=5, help="Number of unfoldings in DeepFRI"
)
parser.add_argument(
    "--hidden", type=int, default=256, help="Number of neurons in each hidden layer"
)
parser.add_argument(
    "--index", type=int, default=0, help="Index for the seperation array"
)
parser.add_argument(
    "--G",
    default=None,
    help="in case the forward matrix needs to be specified",
)
parser.add_argument(
    "--use_emoms", type=bool, default=False, help="If the sampling kernel is EMOMS"
)
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
a_data = torch.tensor(a_kmax, dtype=torch.float64)
dset = torch.utils.data.TensorDataset(Y_data, x_data, a_data)

Y = dset[:][0]
ak_max = dset[:][2]

Y = Y[:1000][:]
ak_max = ak_max[:1000]
# print(ak_max.dtype)

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

params.G = forward_mtx

if params.use_emoms:
    cmtx = loadmat(os.path.join(parent_dir, "data/emoms_data/c_mn.mat"))
    params.C = cmtx["c_mn"]
    model = DeepFRI_EMOMS(params)
else:
    model = DeepFRI(params)

if params.K == 2:
    if params.use_emoms:
        model_file = f"results/K{params.K}/models/model_EMOMS_{seperations[params.index]:0.5f}_N{params.N}_K{params.K}_gamma{params.gamma}.pth"
    else:
        model_file = f"results/K{params.K}/models/model_RMD{seperations[params.index]:0.5f}_N{params.N}_K{params.K}_gamma{params.gamma}.pth"
    model.load_state_dict(torch.load(os.path.join(parent_dir, model_file)))
    model.eval()

print("####################### Testing deepfri ####################### ")
for i, psnr in enumerate(psnrs):

    if params.K != 2:
        model_file = f"results/K{params.K}/models/model_PSNR{psnr}_N{params.N}_K{params.K}_gamma{params.gamma}.pth"
        model.load_state_dict(torch.load(os.path.join(parent_dir, model_file)))
        model.eval()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

    Y.to(dtype=torch.complex128)

    print(f"PSNR={np.floor(psnr)}")
    noise = gen_noise_psnr(Y, ak_max, psnr)
    Y_noisy = Y + noise

    sig = compute_sigma_psnr(ak_max, psnr)

    x_hat = model(Y_noisy, sig)

    fourier_coeffs = x_hat.detach().numpy()

    results = Parallel(n_jobs=-1)(
        delayed(test_deepfri)(
            i, fourier_coeffs, t, params.M, params.T, use_emoms=params.use_emoms
        )
        for i in tqdm(range(1000))
    )

    results = np.array(results)

    errors.append(results)

errors = np.array(errors)

results_dir = f"results/K{params.K}/plots/deepfri"
save_dir = os.path.join(parent_dir, results_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if params.K == 2:
    if params.use_emoms:
        filename = f"deepfri_EMOMS_N{params.N}_K{params.K}_gamma{params.gamma}_{seperations[params.index]:0.5f}.npy"
    else:
        filename = f"deepfri_N{params.N}_K{params.K}_gamma{params.gamma}_{seperations[params.index]:0.5f}.npy"
else:
    filename = f"deepfri_N{params.N}_K{params.K}_gamma{params.gamma}.npy"

np.save(os.path.join(save_dir, filename), errors)

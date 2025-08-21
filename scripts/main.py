"""

DeepFRI NETWORK TRAINING CODE
Author: Sharan Patil, Indian Institute of Science

"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils import *
from src.lib import *
from src.dataset import *
from tqdm import tqdm

import argparse

exponents = np.linspace(-3, -1, num=9)
seperations = 10**exponents

parser = argparse.ArgumentParser("Training DeepFRI")

parser.add_argument("--N", type=int, default=21, help="Number of samples")
parser.add_argument(
    "--K", type=int, default=2, help="Number of teeth in the dirac comb"
)
parser.add_argument("--gamma", type=int, default=5, help="Oversampling Parameter")
parser.add_argument("--T", type=float, default=1.0, help="Time Period [0, T]")
parser.add_argument(
    "--num_unfoldings", type=int, default=5, help="Number of unfoldings in DeepFRI"
)
parser.add_argument(
    "--hidden", type=int, default=256, help="Number of neurons in each hidden layer"
)
parser.add_argument("--depth", type=int, default=2, help="Depth of the denoiser")
parser.add_argument(
    "--psnr", type=float, default=0.0, help="psnr level of noisy samples"
)
parser.add_argument("--mode", type=str, default="train", help="train/test")
parser.add_argument(
    "--psnr_range", type=float, nargs="*", help="psnr range of samples, K2 case"
)
parser.add_argument("--index", type=int, default=0, help="K2 synthetic")
parser.add_argument(
    "--ultrasound", type=bool, default=False, help="train on ultrasound synthetic data"
)
parser.add_argument(
    "--G",
    default=None,
    help="in case the forward matrix needs to be specified",
)

params = parser.parse_args()
params.M = params.K * params.gamma
params.sampling_times = np.arange(0, params.N) * params.T / params.N

## Train on gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)

##  Load the training data
if params.K == 2:
    exponents = np.linspace(-3, -1, num=9)
    seperations = 10**exponents
    if not params.ultrasound:
        print(f"Seperation selected: {seperations[params.index]:0.5f}")

train_data, a, t, parent_dir = load_data(
    params, mode=params.mode, seperations=seperations, ultrasound=params.ultrasound
)

## Extract samples (Y_train), fourier coefficients (x_train) and amplitudes (a) from the training set
Y_train = train_data[:][0]
x_train = train_data[:][1]

## Store the maximum amplitudes of the signals
a_kmax = np.max(np.abs(a), axis=1)
dset = torch.utils.data.TensorDataset(
    Y_train, x_train, torch.tensor(a_kmax, dtype=torch.float64)
)

## Instantiate the DeepFRI model
fry = DeepFRI(params)
fry.to(device=device)

## Set the optimiser and learning rate scheduler
optim_net = torch.optim.Adam(fry.parameters(), lr=8e-4, weight_decay=1e-7)
scheduler = torch.optim.lr_scheduler.StepLR(optim_net, step_size=60, gamma=0.1)

train_loader = torch.utils.data.DataLoader(dset, batch_size=Y_train.shape[0] // 10)
loss_fn = nn.MSELoss()

print("Parameters: ", params)
print("Training start")
fry.train()

psnr = params.psnr
psnr_range = params.psnr_range
num_epochs = 200
loss_train = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for Y, x, akmax in train_loader:
        Y = Y.to(device=device)
        x = x.to(device=device)

        if params.K == 2:
            ## train on a range of PSNRs for all samples in a batch
            psnr_vec = psnr_range[0] + torch.rand(Y.shape[0]) * (
                psnr_range[-1] - psnr_range[0]
            )
        else:
            ## single PSNR for all samples in a batch
            psnr_vec = psnr * torch.ones(Y.shape[0])

        ## noise level is required by the denoising network
        noise_level = ((10 ** (-psnr_vec / 20)) * akmax)[:, None]
        noise = gen_noise_psnr(Y, akmax, psnr_vec)

        Y_noisy = Y + noise.to(device=Y.device)
        x_hat = fry(Y_noisy, noise_level.to(Y.device))

        loss_real = loss_fn(x_hat.real, x.real)
        loss_imag = loss_fn(x_hat.imag, x.imag)
        loss_all = loss_real + loss_imag
        epoch_loss += loss_all.item()

        loss_all.backward()
        optim_net.step()
        optim_net.zero_grad()
    scheduler.step()

    print(
        f"epoch {epoch + 1} \t LR {scheduler.get_last_lr()[0]} \t loss {epoch_loss:0.4f} \t tau {fry.tau:0.4f}"
    )
    loss_train.append(epoch_loss)

results_dir = f"results/K{params.K}/models"
save_dir = os.path.join(parent_dir, results_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if params.K == 2:
    ## individual models are trained on a range of PSNRs for on fixed seperation distribution (see training scheme in the paper)
    filename = f"model_RMD{seperations[params.index]:0.5f}_N{params.N}_K{params.K}_gamma{params.gamma}.pth"
else:
    ## individual models are trained on a range of seperations for a fixed PSNR level
    filename = (
        # f"model_PSNR{params.psnr}_N{params.N}_K{params.K}_gamma{params.gamma}.pth"
        f"model_PSNR{params.psnr}_N{params.N}_K{params.K}_gamma{params.gamma}_hidden{params.hidden}_depth{params.depth}.pth"
    )

## Save the model
torch.save(fry.state_dict(), os.path.join(save_dir, filename))

"""

NEURAL NETWORK COMPONENTS FOR DeepFRI
Author: Sharan Patil, Indian Institute of Science

"""

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from src.fri_algorithms import cadzow_ls_torch
from complexPyTorch.complexLayers import ComplexLinear

import matplotlib.pyplot as plt

class ComplexSoftShrink(nn.Module):
    """A noise-level dependent softshrink activation layer

    Threshold parameter is directly proportional to the noise level of the
    samples.

    Attributes:
        k (torch.nn.Parameter): learnable parameter which controls the threshold parameter of the softshrink activation
    """

    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor([0.005]), requires_grad=False)

    def set_lambda(self, sigma):
        """Set the threshold parameter of the softshrink activation

        Arguments:
            sigma (float): the noise variance

        Returns:
            None
        """
        self.lamda = self.k * (sigma)

    def softshrink(self, x):
        """Compute the softshrink activation function of a given input

        Arguments:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: soft thresholded x
        """

        zero = torch.zeros(x.shape).to(x.device)

        return torch.maximum(x - self.lamda, zero) - torch.maximum(
            -x - self.lamda, zero
        )

    def forward(self, x):
        """Forward pass

        Arguments:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: soft-thresholded tensor
        """
        return self.softshrink(x.real) + 1j * self.softshrink(x.imag)


class DenoisingBlock(nn.Module):
    """Denoising layer after the gradient step

    Attributes:
        activation: complex softshrink activation function
        layers (torch.nn.ModuleList): complex linear layers that make up the denoiser network
    """

    def __init__(self, params):
        """
        Arguments:
            params (Namespace): DeepFRI parameter space
        Returns:
            None
        """

        super().__init__()

        self.activation = ComplexSoftShrink()
        self.layers = torch.nn.ModuleList()
        self.layers.append(ComplexLinear(2 * params.M + 1, params.hidden))
        for _ in range(params.depth):
            self.layers.append(ComplexLinear(params.hidden, params.hidden))
        self.layers.append(ComplexLinear(params.hidden, 2 * params.M + 1))

    def forward(self, x):
        """Denoiser network forward pass

        Arguments:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: denoised tensor
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)

        return x


class DeepFRI(nn.Module):
    """A deep unfolded network for FRI signal reconstruction

    The samples of y can be written in matrix-vector form as:

        yn = G @ (xm + w)   where G is the forward matrix
                                    xm is a vector of fourier coefficients of x
                                    w is white gaussian noise

    Attributes:
        K (int): number of diracs in the ground-truth signal
        M (int): dimension
        N (int): the number of samples of y
        T (float): period of the ground-truth signal
        num_unfoldings (int): number of deep unfoldings in the network
        G_np (np.ndarray): the forward matrix
        G (torch.Tensor): transpose of the forward matrix
        tau: step size in the gradient descent step, learnable
        blocks (torch.nn.ModuleList): a module list containing the denoising
                                      network blocks
    """

    def __init__(self, params):
        super().__init__()

        self.K = params.K
        self.M = params.K * params.gamma
        self.P = self.M
        self.N = params.N
        self.T = params.T
        self.num_unfoldings = params.num_unfoldings

        if params.G is None:
            self.G_np = np.exp(
                1j
                * 2
                * np.pi
                * np.outer(
                    np.arange(0, self.N, dtype=float) * self.T / self.N,
                    np.arange(-self.M, self.M + 1),
                )
                / self.T
            )
        else:
            self.G_np = params.G

        self.Gpinv = torch.tensor(np.linalg.pinv(self.G_np), dtype=torch.complex128)

        self.G = torch.nn.Parameter(
            torch.tensor(self.G_np.T, dtype=torch.complex128),
        )

        self.tau = 1 - (1 / (np.sqrt(self.P + 1)))
        self.block = DenoisingBlock(params)

    def forward(self, y, sig):
        x = torch.zeros((y.shape[0], 2 * self.M + 1), dtype=torch.complex128).to(
            y.device
        )
        target = y @ torch.t(self.Gpinv.to(y.device))
        zeta = 7
        alpha = (1 - torch.exp(-zeta * sig)) / (1 + torch.exp(-zeta * sig))
        # alpha = 1

        if self.training:
            for _ in range(self.num_unfoldings):
                self.block.activation.set_lambda(sig)
                der = x - target
                z = x - self.tau * der
                x = self.block(z)
        else:
            print("Testing DeepFRI...")
            for i in tqdm(range(self.num_unfoldings)):
                self.block.activation.set_lambda(sig)

                der = x - target
                z = x - (self.tau) * der
                fri = self.block(z)
                cdz = cadzow_ls_torch(z, self.M, self.P, self.K)
                scale = (torch.linalg.norm(cdz, dim=1) / torch.linalg.norm(fri, dim=1))[
                    :, None
                ]
                x = alpha * (scale) * fri + (1 - alpha) * (cdz)
        return x


class DeepFRI_EMOMS(nn.Module):
    """A deep unfolded network for FRI signal reconstruction

    The samples of y can be written in matrix-vector form as:

        yn = G @ (xm + w)   where G is the forward matrix
                                    xm is a vector of fourier coefficients of x
                                    w is white gaussian noise

    Attributes:
        K (int): number of diracs in the ground-truth signal
        M (int): dimension
        N (int): the number of samples of y
        T (float): period of the ground-truth signal
        num_unfoldings (int): number of deep unfoldings in the network
        G_np (np.ndarray): the forward matrix
        G (torch.Tensor): transpose of the forward matrix
        tau: step size in the gradient descent step, learnable
        blocks (torch.nn.ModuleList): a module list containing the denoising
                                      network blocks
    """

    def __init__(self, params):
        super().__init__()

        self.K = params.K
        self.M = params.K * params.gamma
        self.P = self.M
        self.N = params.N
        self.T = params.T
        self.num_unfoldings = params.num_unfoldings

        self.C_np = params.C
        self.C = torch.nn.Parameter(
            torch.tensor(self.C_np.T, dtype=torch.complex128), requires_grad=False
        )

        self.tau = 1 - (1 / (2 * np.sqrt(self.P + 1)))
        self.block = DenoisingBlock(params)

    def forward(self, y, sig):
        s = torch.zeros((y.shape[0], 2 * self.M + 1), dtype=torch.complex128).to(
            y.device
        )
        target = y @ self.C
        zeta = 7
        alpha = (1 - torch.exp(-zeta * sig)) / (1 + torch.exp(-zeta * sig))

        if self.training:
            for _ in range(self.num_unfoldings):
                self.block.activation.set_lambda(sig)
                der = s - target
                z = s - self.tau * der
                s = self.block(z)
        else:
            print("Testing DeepFRI...")
            for i in tqdm(range(self.num_unfoldings)):
                self.block.activation.set_lambda(sig)

                der = s - target
                z = s - (self.tau) * der
                fri = self.block(z)
                cdz = cadzow_ls_torch(z, self.M, self.P, self.K)
                scale = (torch.linalg.norm(cdz, dim=1) / torch.linalg.norm(fri, dim=1))[
                    :, None
                ]
                s = alpha * (scale) * fri + (1 - alpha) * (cdz)
        return s

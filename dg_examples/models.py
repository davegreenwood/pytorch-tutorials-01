"""Encoders, Decoders, Discriminators. Simple models to import."""

import torch


# -----------------------------------------------------------------------------
# LOSS Functions
# -----------------------------------------------------------------------------


class DiscrimL2Loss(torch.nn.Module):
    """ Real, Fake.

    Real and fake are in range 0 to 1. They are output from a discriminator.
    As discriminator improves, real should tend toward
    1, and fake to 0.
    https://arxiv.org/abs/1611.04076v2
    D_loss = 0.5 * (torch.mean((D_real - 1)**2) + torch.mean(D_fake**2))
    """

    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        return 0.5 * (torch.mean((real - 1) ** 2) + torch.mean(fake ** 2))


class DiscrimNegLogLoss(torch.nn.Module):
    """ Real, Fake.

    Real and fake are in range 0 to 1. They are output from a discriminator.
    As discriminator improves, real should tend toward
    1, and fake to 0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        eps = 1e-9
        return - (torch.log(eps + real) + torch.log(eps + 1 - fake)).mean()


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------


class MLPEncoder(torch.nn.Module):
    """A simple 3-layer MLP encoder."""
    def __init__(self, xdim, zdim=12, hidden=128):
        super().__init__()
        self.one = torch.nn.Linear(xdim, hidden)
        self.two = torch.nn.Linear(hidden, hidden)
        self.three = torch.nn.Linear(hidden, zdim)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """Override the forward method of torch.nn.Module"""
        x = self.one(x)
        x = self.relu(x)
        x = self.two(x)
        x = self.relu(x)
        x = self.three(x)
        return x


class MLPDecoder(torch.nn.Module):
    """A simple 3-layer MLP encoder."""
    def __init__(self, xdim, zdim=12, hidden=128):
        super().__init__()
        self.one = torch.nn.Linear(zdim, hidden)
        self.two = torch.nn.Linear(hidden, hidden)
        self.three = torch.nn.Linear(hidden, xdim)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """Override the forward method of torch.nn.Module"""
        x = self.one(x)
        x = self.relu(x)
        x = self.two(x)
        x = self.relu(x)
        x = self.three(x)
        x = self.relu(x)
        return x


class MLPDiscriminator(torch.nn.Module):
    """A simple 3-layer MLP discriminator.

    This example uses Sequential, and calls the resulting model in forward.
    """
    def __init__(self, zdim=12, hidden=128):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(zdim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

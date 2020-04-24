"""Testing different discriminators NLL Loss"""

# %%

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


from dg_examples.data import MNISTMLP
from dg_examples.models import (
    MLPDecoder, MLPEncoder, MLPDiscriminator, DiscrimNegLogLoss)


# -----------------------------------------------------------------------------
# HYPER-PARAMS
# -----------------------------------------------------------------------------

XDIM = 28 * 28
ZDIM = 2
BATCH = 100
EPOCHS = 100
LR = 0.001

# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)

data = MNISTMLP(batch_size=BATCH)
train, test = data.get_loaders()

encoder = MLPEncoder(xdim=XDIM, zdim=ZDIM).to(DEVICE)
decoder = MLPDecoder(xdim=XDIM, zdim=ZDIM).to(DEVICE)
discrim = MLPDiscriminator(zdim=ZDIM).to(DEVICE)

gan_loss_fnc = DiscrimNegLogLoss()
rcn_loss_fnc = torch.nn.MSELoss()

optim_gen = torch.optim.Adam(encoder.parameters(), lr=LR)
optim_dis = torch.optim.Adam(discrim.parameters(), lr=LR)
optim_rcn = torch.optim.Adam(
    [p for p in encoder.parameters()] +
    [p for p in decoder.parameters()], lr=LR)
opt = [optim_gen, optim_rcn, optim_dis]

# %%

WRITER = SummaryWriter(os.path.expanduser("~/runs/mnist/NLL"))
# in a terminal run:
# tensorboard --logdir=~/runs/mnist

k = 0
for epoch in range(EPOCHS):
    for i, (x, _) in enumerate(train):
        x = x.to(DEVICE)
        # reset optimiser
        [o.zero_grad() for o in opt]

        # latent space
        z_fake = encoder(x)

        # reconstruction
        x_rcn = decoder(z_fake)
        r_loss = rcn_loss_fnc(x, x_rcn)
        r_loss.backward()
        optim_rcn.step()

        # discrim
        d_fake = discrim(z_fake.detach())
        d_real = discrim(torch.randn(BATCH, ZDIM, device=DEVICE))
        d_loss = gan_loss_fnc(d_real, d_fake)
        d_loss.backward()
        optim_dis.step()

        # generator
        d_fake = discrim(encoder(x))
        g_loss = - torch.log(d_fake + 1e-9).mean()
        g_loss.backward()
        optim_gen.step()

        # logging - the log file can get very large so only log 1/100
        if k % 100 == 0:
            x_grid = make_grid(x.reshape(BATCH, 1, 28, 28), nrow=10)
            x_rcn_grid = make_grid(x_rcn.reshape(BATCH, 1, 28, 28), nrow=10)
            grid = torch.cat([x_grid, x_rcn_grid], dim=2)
            WRITER.add_scalar(f"Loss/d_loss", d_loss.item(), k)
            WRITER.add_scalar(f"Loss/r_loss", r_loss.item(), k)
            WRITER.add_scalar(f"Loss/g_loss", g_loss.item(), k)
            WRITER.add_image("Image/x", grid, k)
            print(f"Evaluation: {k}")
        k += 1


# -----------------------------------------------------------------------------
# SAVING
# -----------------------------------------------------------------------------

SAVE = "./saved_models/"
os.makedirs(SAVE, exist_ok=True)
torch.save(encoder, SAVE + "mnist_encoder_02.pkl")
torch.save(decoder, SAVE + "mnist_decoder_02.pkl")
torch.save(discrim, SAVE + "mnist_discrim_02.pkl")

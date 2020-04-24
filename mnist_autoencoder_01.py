"""Testing different discriminators """

# %%

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


from dg_examples.data import MNISTMLP
from dg_examples.models import (
    MLPDecoder, MLPEncoder, MLPDiscriminator, DiscrimL2Loss)


# -----------------------------------------------------------------------------
# HYPER-PARAMS
# -----------------------------------------------------------------------------

XDIM = 28 * 28
ZDIM = 2
BATCH = 100
EPOCHS = 100
LR = 0.0001

# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------

data = MNISTMLP(batch_size=BATCH)
train, test = data.get_loaders()

encoder = MLPEncoder(xdim=XDIM, zdim=ZDIM)
decoder = MLPDecoder(xdim=XDIM, zdim=ZDIM)
discrim = MLPDiscriminator(zdim=ZDIM)

gan_loss_fnc = DiscrimL2Loss()
rcn_loss_fnc = torch.nn.MSELoss()

# optimise all-in-one
optim = torch.optim.Adam(
    [p for p in encoder.parameters()] +
    [p for p in decoder.parameters()] +
    [p for p in discrim.parameters()], lr=LR
)


# %%

WRITER = SummaryWriter(os.path.expanduser("~/runs/mnist/L2"))
# in a terminal run:
# tensorboard --logdir=~/runs/mnist

k = 0
for epoch in range(EPOCHS):
    for i, (x, _) in enumerate(train):
        # reset optimiser
        optim.zero_grad()

        # latent space
        z_fake = encoder(x)
        z_real = torch.randn_like(z_fake)

        # discrim
        d_fake = discrim(z_fake)
        d_real = discrim(z_real)

        # reconstruction
        x_rcn = decoder(z_fake)

        # losses
        d_loss = gan_loss_fnc(d_real, d_fake)
        r_loss = rcn_loss_fnc(x, x_rcn)
        loss = d_loss + r_loss

        # update
        loss.backward()
        optim.step()

        # logging - the log file can get very large so only log 1/100
        if k % 100 == 0:
            x_grid = make_grid(x.reshape(BATCH, 1, 28, 28), nrow=10)
            x_rcn_grid = make_grid(x_rcn.reshape(BATCH, 1, 28, 28), nrow=10)
            grid = torch.cat([x_grid, x_rcn_grid], dim=2)
            WRITER.add_scalar(f"Loss/d_loss", d_loss.item(), k)
            WRITER.add_scalar(f"Loss/r_loss", r_loss.item(), k)
            WRITER.add_scalar(f"Loss/comb_loss", loss.item(), k)
            WRITER.add_image("Image/x", grid, k)
            print(f"Evaluation: {k}")
        k += 1


# -----------------------------------------------------------------------------
# SAVING
# -----------------------------------------------------------------------------

SAVE = "./saved_models/"
os.makedirs(SAVE, exist_ok=True)
torch.save(encoder, SAVE + "mnist_encoder_01.pkl")
torch.save(decoder, SAVE + "mnist_decoder_01.pkl")
torch.save(discrim, SAVE + "mnist_discrim_01.pkl")

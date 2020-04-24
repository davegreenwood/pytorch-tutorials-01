"""Testing different discriminators """

# %%

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


from dg_examples.data import MNISTMLP
from dg_examples.models import (MLPDecoder, MLPEncoder)


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

rcn_loss_fnc = torch.nn.MSELoss()

# optimise all-in-one
optim = torch.optim.Adam(
    [p for p in encoder.parameters()] +
    [p for p in decoder.parameters()], lr=LR
)


# %%

WRITER = SummaryWriter(os.path.expanduser("~/runs/mnist/PL"))
# in a terminal run:
# tensorboard --logdir=~/runs/mnist

k = 0
for epoch in range(EPOCHS):
    for i, (x, _) in enumerate(train):
        x = x.to(DEVICE)
        # reset optimiser
        optim.zero_grad()

        # latent space
        z_fake = encoder(x)

        # reconstruction
        x_rcn = decoder(z_fake)

        # losses
        loss = rcn_loss_fnc(x, x_rcn)

        # update
        loss.backward()
        optim.step()

        # logging - the log file can get very large so only log 1/100
        if k % 100 == 0:
            x_grid = make_grid(x.reshape(BATCH, 1, 28, 28), nrow=10)
            x_rcn_grid = make_grid(x_rcn.reshape(BATCH, 1, 28, 28), nrow=10)
            grid = torch.cat([x_grid, x_rcn_grid], dim=2)
            WRITER.add_scalar(f"Loss/r_loss", loss.item(), k)
            WRITER.add_image("Image/x", grid, k)
            print(f"Evaluation: {k}")
        k += 1


# -----------------------------------------------------------------------------
# SAVING
# -----------------------------------------------------------------------------

SAVE = "./saved_models/"
os.makedirs(SAVE, exist_ok=True)
torch.save(encoder, SAVE + "mnist_encoder_00.pkl")
torch.save(decoder, SAVE + "mnist_decoder_00.pkl")

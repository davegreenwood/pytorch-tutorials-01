"""Testing different discriminators """

# %%

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dg_examples.utils import mnist_label_images, accuracy
from dg_examples.data import MNISTMLP
from dg_examples.models import CNNClassifier


# -----------------------------------------------------------------------------
# HYPER-PARAMS
# -----------------------------------------------------------------------------

SIZE = (1, 28, 28)
NCLASS = 10
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

model = CNNClassifier(image_size=SIZE, nclasses=NCLASS).to(DEVICE)

optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()


# %%

WRITER = SummaryWriter(os.path.expanduser("~/runs/mnist/cnn"))
# in a terminal run:
# tensorboard --logdir=~/runs/mnist/cnn

k = 0
for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(train):

        x = x.view(-1, 1, 28, 28).to(DEVICE)
        y = y.to(DEVICE)

        # reset optimiser
        optim.zero_grad()

        # reconstruction
        logits = model(x)

        # losses
        loss = loss_func(logits, y)

        # update
        loss.backward()
        optim.step()

        # logging - the log file can get very large so only log 1/100
        if k % 100 == 0:
            x_grid = make_grid(x.reshape(BATCH, 1, 28, 28), nrow=10)
            y_grid = make_grid(mnist_label_images(y), nrow=10)
            grid = torch.cat([x_grid, y_grid], dim=2)
            WRITER.add_scalar(f"Loss/r_loss", loss.item(), k)
            WRITER.add_image("Image/x", grid, k)
            print(f"Evaluation: {k}")
        k += 1

    correct, total = accuracy(test, model, DEVICE, view=(-1, 1, 28, 28))
    WRITER.add_scalar(f"Accuracy", (100 * correct / total), k)

# -----------------------------------------------------------------------------
# SAVING
# -----------------------------------------------------------------------------

SAVE = "./saved_models/"
os.makedirs(SAVE, exist_ok=True)
torch.save(model, SAVE + "mnist_classifier_01.pkl")


# %%

"""utilities for plotting etc."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas


kelly_colors = [
    "#F2F3F4",
    "#222222",
    "#F3C300",
    "#875692",
    "#F38400",
    "#A1CAF1",
    "#BE0032",
    "#C2B280",
    "#848482",
    "#008856",
    "#E68FAC",
    "#0067A5",
    "#F99379",
    "#604E97",
    "#F6A600",
    "#B3446C",
    "#DCD300",
    "#882D17",
    "#8DB600",
    "#654522",
    "#E25822",
    "#2B3D26",
]


def mnist_latent(encoder, test, device):
    """test is the test dataloader"""

    a, b = zip(*[i for i in test])
    data = encoder(torch.cat(a, dim=0).to(device)).detach().cpu()
    label = torch.cat(b)

    fig, ax = plt.subplots(figsize=[8, 8], dpi=72)

    for i, c in enumerate(kelly_colors[2:12]):
        idx = label == i
        ax.plot(data[idx, 0], data[idx, 1], "o", c=c, label=str(i))
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.legend(loc=1)

    plt.tight_layout()
    plt.close(fig)
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    return torch.tensor(img[..., :3]).permute(2, 0, 1)

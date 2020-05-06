"""utilities for plotting etc."""

import torch
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas


TOTEN = ToTensor()


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


def accuracy(testloader, model, device):
    """Predict the test et accuracy."""
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


def mnist_label_image(label):
    device = label.device
    img = Image.new('RGB', (28, 28), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10, 5), str(int(label)), fill=(255, 255, 0))
    return TOTEN(img).to(device)


def mnist_label_images(labels):
    return [mnist_label_image(label) for label in labels]


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

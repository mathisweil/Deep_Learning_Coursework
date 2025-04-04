from IPython import display
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms
from matplotlib_inline import backend_inline
from torchvision.transforms import TrivialAugmentWide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
scaler = torch.GradScaler('cuda')

# Data augmentation for training and testing.
train_augmentations = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])
])

test_augmentations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])
])


def use_svg_display() -> None:
    """Use the SVG format to display plots in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')


def set_axes(axes: plt.Axes, xlabel: str, ylabel: str, xlim, ylim,
             xscale: str, yscale: str, legend: list) -> None:
    """Configure the axes for a matplotlib plot."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def get_dataloader_workers() -> int:
    """Return the number of subprocesses to use for data loading."""
    return 10


def load_data_cifar10(batch_size: int, resize: int = None):
    """
    Download the CIFAR-10 dataset and return training and testing DataLoaders.

    :param batch_size: Size of each batch.
    :param resize: Optional parameter for resizing images (not used here).
    :return: Tuple of (train_loader, test_loader)
    """
    cifar10_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=train_augmentations, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=test_augmentations, download=True)

    generator = torch.Generator(device='cpu')

    train_loader = data.DataLoader(
        cifar10_train, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=get_dataloader_workers(), pin_memory=True, persistent_workers=True)

    test_loader = data.DataLoader(
        cifar10_test, batch_size=batch_size, shuffle=False, generator=generator,
        num_workers=get_dataloader_workers(), pin_memory=True, persistent_workers=True)

    return train_loader, test_loader


class Animator:
    """Class for animating plots during training."""

    def __init__(self, xlabel: str = None, ylabel: str = None, legend: list = None,
                 xlim=None, ylim=None, xscale: str = 'linear', yscale: str = 'linear',
                 fmts: tuple = ('-', 'm--', 'g-.', 'r:'), nrows: int = 1, ncols: int = 1,
                 figsize: tuple = (3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

        self.display_handle = display.display(self.fig, display_id=True)

    def add(self, x, y) -> None:
        """Add a new data point and update the plot."""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        self.display_handle.update(self.fig)
        # display.display(self.fig)
        # display.clear_output(wait=True)


class Accumulator:
    """Accumulate sums over multiple variables."""

    def __init__(self, n: int):
        self.data = [0.0] * n

    def add(self, *args) -> None:
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(torch.sum(cmp))


def evaluate_accuracy(net, data_iter, device=None) -> float:
    """
    Compute the accuracy of a model on a dataset.

    :param net: The neural network model.
    :param data_iter: DataLoader for the evaluation dataset.
    :param device: The device to use.
    :return: Accuracy as a float.
    """
    if isinstance(net, nn.Module):
        net.eval()
        if device is None:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device, non_blocking=True) for x in X]
            else:
                X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            metric.add(accuracy(net(X), y), y.size(0))
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, optimizer):
    """
    Train the model for one epoch.

    :return: Tuple (train loss, train accuracy)
    """
    net.train()
    metric = Accumulator(3)
    progress_bar = tqdm(train_iter, desc="Training", leave=False)

    for X, y in progress_bar:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast('cuda'):
            y_hat = net(X)
            l = loss(y_hat, y)
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
        metric.add(l.item() * y.size(0), accuracy(y_hat, y), y.numel())

        progress_bar.set_postfix({
            "loss": f"{metric[0] / metric[2]:.4f}",
            "acc": f"{metric[1] / metric[2]:.4f}"
        })
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs: int, optimizer, scheduler,
          checkpoint_path: str = 'checkpoints'):
    """
    Train a model and save checkpoints periodically with TensorBoard logging.

    :param net: Neural network model.
    :param train_iter: Training DataLoader.
    :param test_iter: Testing DataLoader.
    :param loss: Loss function.
    :param num_epochs: Total number of epochs.
    :param optimizer: Optimiser.
    :param scheduler: Learning rate scheduler.
    :param checkpoint_path: Directory to save checkpoints.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    start_epoch = 0

    latest_ckpt = os.path.join(checkpoint_path, 'latest.pth')
    if os.path.exists(latest_ckpt):
        print(f"Loading checkpoint from {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter, device)

        animator.add(epoch + 1, (train_loss, train_acc, test_acc))

        scheduler.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, f'epoch_{epoch + 1}.pth'))
            torch.save(checkpoint, latest_ckpt)

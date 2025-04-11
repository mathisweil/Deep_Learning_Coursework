import os

from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms

# Set CUDA memory allocation configuration to limit the maximum split size.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Determine the device: use CUDA if available, otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable the cudnn benchmark mode for potential performance improvements.
torch.backends.cudnn.benchmark = True

# Initialise the gradient scaler for mixed-precision training.
scaler = torch.GradScaler('cuda')

# =============================================================================
# Data Augmentation and Transformation Pipelines
# =============================================================================
train_augmentations = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010]),
    transforms.RandomErasing(p=0.25)
])

test_augmentations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])
])

# =============================================================================
# Helper Functions for Visualization and Data Loading
# =============================================================================
def use_svg_display() -> None:
    """Configure matplotlib to display plots in SVG format within Jupyter."""
    backend_inline.set_matplotlib_formats('svg')


def set_axes(axes: plt.Axes, xlabel: str, ylabel: str, xlim, ylim,
             xscale: str, yscale: str, legend: list, yticks=None) -> None:
    """
    Configure the axes of a matplotlib plot.

    Args:
        axes (plt.Axes): The axes to configure.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        xlim: Limits for the X-axis.
        ylim: Limits for the Y-axis.
        xscale (str): Scale for the X-axis ('linear', 'log', etc.).
        yscale (str): Scale for the Y-axis ('linear', 'log', etc.).
        legend (list): List of legend entries.
        yticks: Optional list of Y-axis tick locations.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if yticks is not None:
        axes.set_yticks(yticks)
    if legend:
        axes.legend(legend)
    axes.grid()


def get_dataloader_workers() -> int:
    """Return the number of subprocesses to use for data loading."""
    return 10  # Adjust this number based on your system's capacity.


def load_data_cifar10(batch_size: int):
    """
    Download the CIFAR-10 dataset and return DataLoaders for training and testing.

    Args:
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        Tuple[data.DataLoader, data.DataLoader]: The training and testing DataLoaders.
    """
    cifar10_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=train_augmentations, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=test_augmentations, download=True)

    generator = torch.Generator(device='cpu')

    train_loader = data.DataLoader(
        cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=get_dataloader_workers(),
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = data.DataLoader(
        cifar10_test,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=get_dataloader_workers(),
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader

# =============================================================================
# Classes for Training Visualisation and Metric Accumulation
# =============================================================================
class Animator:
    """Class to animate training progress through matplotlib plots."""

    def __init__(self, xlabel: str = None, ylabel: str = None, legend: list = None,
                 xlim=None, ylim=None, xscale: str = 'linear', yscale: str = 'linear',
                 fmts: tuple = ('-', 'm--', 'g-.', 'r:'), nrows: int = 1, ncols: int = 1,
                 figsize: tuple = (3.5, 2.5), yticks=None):
        legend = legend or []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend, yticks
        )
        self.X, self.Y, self.fmts = None, None, fmts
        self.display_handle = display.display(self.fig, display_id=True)

    def add(self, x, y) -> None:
        """
        Add new data points to the plot and update the display.

        Args:
            x: X-axis data point(s).
            y: Corresponding Y-axis data point(s).
        """
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()  # Clear the current axes.
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        self.display_handle.update(self.fig)


class Accumulator:
    """Accumulate sums over multiple variables, e.g., for metrics computation."""

    def __init__(self, n: int):
        self.data = [0.0] * n

    def add(self, *args) -> None:
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

# =============================================================================
# Accuracy and Loss Evaluation Functions
# =============================================================================
def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculate the number of correct predictions.

    Args:
        y_hat (torch.Tensor): The predicted outputs.
        y (torch.Tensor): The true labels.

    Returns:
        float: The count of correct predictions.
    """
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(torch.sum(cmp))


def evaluate_accuracy(net, data_iter, device=None) -> float:
    """
    Evaluate the accuracy of the model over a dataset.

    Args:
        net: The neural network model.
        data_iter: DataLoader for the evaluation dataset.
        device: The device on which computation is performed.

    Returns:
        float: The computed accuracy.
    """
    if isinstance(net, nn.Module):
        net.eval()
        device = device or next(iter(net.parameters())).device
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


def evaluate_loss(net, data_iter, loss, device=None) -> float:
    """
    Evaluate the average loss of the model over a dataset.

    Args:
        net: The neural network model.
        data_iter: DataLoader for the evaluation dataset.
        loss: Loss function to be used.
        device: The computation device.

    Returns:
        float: The average loss.
    """
    if isinstance(net, nn.Module):
        net.eval()
        device = device or next(iter(net.parameters())).device
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device, non_blocking=True) for x in X]
            else:
                X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = net(X)
            l = loss(y_hat, y)
            total_loss += l.item() * y.size(0)
            total_count += y.size(0)
    return total_loss / total_count

# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(net, train_iter, loss, optimizer):
    """
    Train the model for a single epoch.

    Args:
        net: The neural network model.
        train_iter: Training DataLoader.
        loss: Loss function.
        optimizer: Optimiser for parameter updates.

    Returns:
        Tuple[float, float]: Average training loss and accuracy.
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
    Train the neural network and periodically save checkpoints.

    Args:
        net: Neural network model.
        train_iter: Training DataLoader.
        test_iter: Testing DataLoader.
        loss: Loss function.
        num_epochs (int): Total number of epochs.
        optimizer: Optimiser.
        scheduler: Learning rate scheduler.
        checkpoint_path (str, optional): Path for saving checkpoints. Defaults to 'checkpoints'.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    start_epoch = 0
    latest_ckpt = os.path.join(checkpoint_path, 'latest.pth')

    # Load checkpoint if available.
    if os.path.exists(latest_ckpt):
        print(f"Loading checkpoint from {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Create Animator instances for loss and accuracy.
    animator_loss = Animator(
        xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], ylim=[0, 2.5],
        legend=['train loss', 'test loss'], figsize=(4, 2.5)
    )
    animator_acc = Animator(
        xlabel='epoch', ylabel='accuracy', xlim=[1, num_epochs], ylim=[0, 1],
        legend=['train acc', 'test acc'], figsize=(4, 3.5),
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    )

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter, device)
        test_loss = evaluate_loss(net, test_iter, loss, device)

        animator_loss.add(epoch + 1, (train_loss, test_loss))
        animator_acc.add(epoch + 1, (train_acc, test_acc))

        scheduler.step()

        # Periodically save checkpoints.
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            ckpt_path = os.path.join(checkpoint_path, f'epoch_{epoch + 1}.pth')
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, latest_ckpt)
        
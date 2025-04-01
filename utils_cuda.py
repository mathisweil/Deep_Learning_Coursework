from IPython import display
from matplotlib import pyplot as plt
import os

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from matplotlib_inline import backend_inline
import multiprocessing
from torchvision.transforms import TrivialAugmentWide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

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


def use_svg_display():  # @save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def get_dataloader_workers():  # @save
    """Use 4 processes to read the data."""
    return 4


def load_data_cifar10(batch_size, resize=None):  # @save
    """Download the CIFAR-10 dataset and load it with augmentations."""
    train_transforms = train_augmentations
    test_transforms = test_augmentations

    cifar10_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=train_transforms, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=test_transforms, download=True)

    generator = torch.Generator(device)
    return (data.DataLoader(cifar10_train, batch_size, shuffle=True, generator=generator,
                            num_workers=get_dataloader_workers(), pin_memory=True),
            data.DataLoader(cifar10_test, batch_size, shuffle=False, generator=generator,
                            num_workers=get_dataloader_workers(), pin_memory=True))


class Animator:  # @save
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
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
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Accumulator:  # @save
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(torch.sum(cmp))


def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    for _, (X, y) in enumerate(data_iter):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# def train_epoch(net, train_iter, loss, updater):  # @save
#     """The training loop for one epoch."""
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         if isinstance(updater, torch.optim.Optimizer):
#             updater.zero_grad()
#             l.backward()
#             updater.step()
#             metric.add(l.item() * y.size(0), accuracy(y_hat, y), y.numel())
#         else:
#             l.sum().backward()
#             updater(X.shape[0])
#             metric.add(l.sum().item(), accuracy(y_hat, y), y.numel())
#     return metric[0] / metric[2], metric[1] / metric[2]


def train_epoch(net, train_iter, loss, optimizer):
    net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y_hat = net(X)
            l = loss(y_hat, y)
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
        metric.add(l.item() * y.size(0), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, optimizer, scheduler, checkpoint_path='checkpoints'):  # @save
    """Train a model and save checkpoints every 10 epochs."""
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
        train_metrics = train_epoch(net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        torch.cuda.empty_cache()

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

    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

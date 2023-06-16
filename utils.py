import torch
import numpy as np
from torchvision import datasets, transforms
import config
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

tqdm = partial(tqdm, position=0, leave=True)


def get_data_loaders():
    kwargs = {"num_workers": config.NUM_WORKERS, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        **kwargs,
    )

    return train_loader, test_loader


class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.device = config.DEVICE

    def train(self, train_loader, test_loader, optimizer, loss_fn, use_scheduler=False):
        for epoch in range(config.EPOCHS):
            print(f"{epoch + 1} / {config.EPOCHS}")
            if use_scheduler:
                adjust_lr(optimizer, epoch)
            self._train(train_loader, optimizer, epoch, loss_fn)
            self._test(test_loader, loss_fn)

    def _train(self, train_loader, optimizer, epoch, loss_fn):
        self.model.train()
        correct_train = 0
        for _, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            self.train_losses.append(loss)
            loss.backward()
            optimizer.step()
            train_pred = output.argmax(dim=1, keepdim=True)
            correct_train += train_pred.eq(target.view_as(train_pred)).sum().item()

            for param in optimizer.param_groups:
                lr = param["lr"]
        print(
            f"""epoch = {epoch + 1} loss={loss.item()}
            train_accuracy={correct_train * 100.0/len(train_loader.dataset)
                            }"""
        )
        self.train_acc.append(correct_train * 100.0 / len(train_loader.dataset))

    def _test(self, test_loader, loss_fn):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += loss_fn(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)
        self.test_acc.append(100.0 * correct / len(test_loader.dataset))

        print(
            """\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n""".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    def run(self, **args):
        self.train(**args)


def adjust_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        init_lr = param_group["lr"]
    lr = max(round(init_lr * 1 / (1 + np.pi / 50 * epoch), 10), 0.0005)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

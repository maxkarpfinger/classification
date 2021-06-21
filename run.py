from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import os
from torchvision import datasets, transforms
from projects.classification.MyModel import MyModel

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

# data
cifar10_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
cifar10_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

#split train into train and validation
cifar10_train, cifar10_val = random_split(cifar10_train, [45000, 5000])

#dataloaders
cifar10_train = DataLoader(cifar10_train, batch_size=64)
cifar10_val = DataLoader(cifar10_val, batch_size=64)
cifar10_test = DataLoader(cifar10_test, batch_size=64)

model = MyModel()
trainer = Trainer(gpus=1)
trainer.fit(model, train_dataloader=cifar10_train, val_dataloaders=cifar10_val)

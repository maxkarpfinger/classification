import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import os
from torchvision import datasets, transforms
from projects.classification.MyModel import MyModel
from optuna.integration import PyTorchLightningPruningCallback
from torch.nn import functional as F
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

# data
cifar10_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
cifar10_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

#split train into train and validation
cifar10_train, cifar10_val = random_split(cifar10_train, [45000, 5000])

#dataloaders
cifar10_train = DataLoader(cifar10_train, batch_size=1024)
cifar10_val = DataLoader(cifar10_val, batch_size=1024)
cifar10_test = DataLoader(cifar10_test, batch_size=1024)


#optuna look for a good learning rate
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    model = MyModel(lr)
    trainer = Trainer(gpus=1, max_epochs=10, callbacks=[EarlyStopping(monitor='val_loss')])
    trainer.fit(model, train_dataloader=cifar10_train, val_dataloaders=cifar10_val)
    losses = []
    for batch in cifar10_val:
        x, y = batch
        logits = model.forward(x)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.cpu().detach().numpy())
    losses = np.asarray(losses)
    loss = np.mean(losses)
    return loss


def run():
    model = MyModel(1e-4)
    trainer = Trainer(gpus=1, max_epochs=200, callbacks=[EarlyStopping(monitor='val_loss')])
    trainer.fit(model, train_dataloader=cifar10_train, val_dataloaders=cifar10_val)


#study = optuna.create_study()
#study.optimize(objective, n_trials=100)

#print(study.best_params)

run()


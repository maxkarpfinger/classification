import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class MyModel(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)
        predictions = torch.argmax(logits, dim=1)
        eq = torch.eq(y, predictions).cpu().detach().numpy()
        accuracy = np.sum(np.where(eq, 1, 0))/np.size(eq)
        self.log('val_acc', accuracy)
        return loss


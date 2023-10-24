import torch
import torch.nn as nn


class Kernel:
    def __init__(self, Model, X, Y) -> None:
        self.Model = Model
        self.lr = 1.0e-04
        self.eps = 0.0
        self.train_loader = None
        self.val_loader = None
        self.optimizer = torch.optim.Adam(Model.parameters(), lr=self.lr, eps=self.eps)
        self.loss_fn = torch.nn.MSELoss()
        self.batch_size = None
        self.num_batches = None
        self.split_train_val(X=X, Y=Y, val_fraction=0.1, batch_size=5)

    def split_train_val(self, X, Y, val_fraction=0.1, batch_size=5) -> None:
        N = X.shape[0]
        N_train = int(N * (1 - val_fraction))
        self.train_loader = torch.utils.data.DataLoader(
            [[X[i], Y[i]] for i in range(N_train)], batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            [[X[i], Y[i]] for i in range(N_train, N)],
            batch_size=batch_size,
            shuffle=True,
        )
        self.batch_size = batch_size
        self.num_batches = int(N_train / batch_size)

    def train_one_epoch(self):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(self.train_loader):
            inputs, labels = data

            self.optimizer.zero_grad()

            outputs = self.Model(inputs)

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            if (i + 1) % self.num_batches == 0:
                last_loss = running_loss / self.batch_size
                running_loss = 0.0

        return last_loss

    def train_model(self, EPOCHS=500):
        epoch_number = 0
        best_vloss = 1.0e06

        for epoch in range(EPOCHS):
            self.Model.train(True)
            avg_loss = self.train_one_epoch()

            running_vloss = 0.0
            self.Model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    vinputs, vlabels = vdata
                    voutputs = self.Model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)

            if (epoch + 1) % (EPOCHS / 10) == 0:
                print("EPOCH {}:".format(epoch + 1))
                print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.Model, "UNet_vcb.pt")

        epoch_number += 1

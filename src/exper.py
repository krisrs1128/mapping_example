#!/usr/bin/env python
from comet_ml import OfflineExperiment
from data import LCMBinary
from train import train, evaluate
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from unet import UNet
import os
import json

data_path = "/data/"
opts = json.load(open("config/defaults.json"))

train_loader = DataLoader(
    LCMBinary(data_path),
    batch_size=opts["batch_size"],
    sampler=SubsetRandomSampler(range(0, 10))
)

test_loader = DataLoader(
    LCMBinary(data_path, "test"),
    batch_size=opts["batch_size"],
    sampler=SubsetRandomSampler(range(0, 10))
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
model = model.to(device)
model.down = [s.to(device) for s in model.down]
model.up = [s.to(device) for s in model.up]
optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
loss_fun = torch.nn.BCEWithLogitsLoss()
loss_fun = loss_fun.to(device)

if __name__ == '__main__':
    exp = OfflineExperiment(
        project_name="lcm_select",
        workspace="krisrs1128",
        offline_directory=os.environ.get("SCRATCH")
    )
    exp.log_parameters(opts)

    # train, and save a model after every 5 epochs
    for epoch in range(opts["n_epochs"]):
        model, train_loss = train(model, train_loader, optimizer, loss_fun, device)
        test_loss = evaluate(model, test_loader, loss_fun, device)

        print(f"Epoch: {epoch} || Train Loss: {train_loss} || Test Loss: {test_loss}")
        exp.log_metrics({
            "train/train_loss": train_loss,
            "train/test_loss": test_loss
        })

        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'model_{}.pt'.format(epoch))

    exp.end()

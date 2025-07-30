# ================================
# train.py
# ================================
import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import CatKeypointDataset
from model import KeypointRegressor

import logging

BATCH_SIZE = 16
EPOCHS = 1
LR = 1e-4
IMAGE_SIZE = 384

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# logging.basicConfig(level=logging.DEBUG)
# logging.debug("++++++++ Setup ++++++++")
# logging.debug(device)

train_dataset = CatKeypointDataset("data/images", "data/labels", image_size=IMAGE_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = KeypointRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        preds = model(images)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "keypoint_model.pth")

# logging.debug("++++++++ End Train ++++++++")
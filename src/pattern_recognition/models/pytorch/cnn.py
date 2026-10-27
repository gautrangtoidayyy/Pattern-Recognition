import time

import torch
import torch.nn as nn


class MNISTCNN(nn.Module):
    """Kiến trúc CNN MNIST chuẩn hoá theo CNN_MNIST_from_scratch_explained.ipynb:
    Input (1, 28, 28) -> Conv2D (32 filters 4x4) -> ReLU -> MaxPool2D (2x2) -> Flatten (4608) -> Dense (128) -> Dense (10)"""
    def __init__(self):
        super().__init__()
        # Conv2D: 32 bộ lọc 4x4, không padding -> shape (28 - 4 + 1) = 25x25
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, padding=0)
        self.relu = nn.ReLU()
        # MaxPool2d: 2x2, stride 2 -> shape floor((25-2)/2)+1 = 12x12
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Dense 128 -> (12 * 12 * 32) * 128 + 128 = 4608 * 128 + 128 = 589,952
        self.fc1 = nn.Linear(12 * 12 * 32, 128)
        # Dense 10 -> 128 * 10 + 10 = 1,290
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predict_cnn(model, X, batch_size=2000):
    model.eval()
    t0 = time.time()
    preds = []
    probs = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size]
            out = model(xb)
            pr = torch.softmax(out, dim=1)
            preds.append(out.argmax(dim=1))
            probs.append(pr)
    t_pred = time.time() - t0
    return torch.cat(preds), torch.cat(probs), t_pred

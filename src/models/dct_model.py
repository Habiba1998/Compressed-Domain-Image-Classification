
import torch
import torch.nn as nn


class DCTClassifier(nn.Module):
    def __init__(self, num_coefficients=1536, num_classes=10):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(num_coefficients, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, 1536)
        return self.classifier(x)

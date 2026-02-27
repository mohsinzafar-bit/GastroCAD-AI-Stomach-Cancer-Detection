import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load pretrained DenseNet201
model = models.densenet201(pretrained=True)

# Modify classifier for binary classification
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Model ready for training")

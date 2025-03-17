import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import uvicorn
from io import BytesIO
import pickle
import logging
import docker

# Logging Setup
logging.basicConfig(level=logging.INFO)

# Configuration
DATASET_PATH = "./data"
MODEL_PATH = "model.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
test_size = int(0.15 * len(dataset))
val_size -= test_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
logging.info("Starting Model Training")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    logging.info(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Save Model
torch.save(model.state_dict(), MODEL_PATH)
logging.info("Model Saved Successfully")

# FastAPI App
app = FastAPI()

def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

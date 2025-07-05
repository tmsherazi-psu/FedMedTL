import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# --------------------------
# Constants
# --------------------------
CATEGORIES = {
    'AK': 0,
    'BCC': 1,
    'BKL': 2,
    'DF': 3,
    'MEL': 4,
    'NV': 5,
    'SCC': 6,
    'VASC': 7
}
NUM_CLASSES = len(CATEGORIES)
IMAGE_SIZE = 224
BATCH_SIZE = 32
CLIENTS = 5
LOCAL_EPOCHS = 5
ROUNDS = 10
DIRICHLET_ALPHA = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Image Preprocessing
# --------------------------

def apply_clahe(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

def Dataset_loader(DIR, RESIZE=224, use_clahe=True):
    IMG = []
    read = lambda imname: np.asarray(cv2.imread(imname))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        if IMAGE_NAME.lower().endswith(('.png', '.jpg', '.jpeg')):
            PATH = os.path.join(DIR, IMAGE_NAME)
            img = read(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
            if use_clahe:
                img = apply_clahe(img)
            IMG.append(np.array(img))
    return np.array(IMG)

# --------------------------
# Data Augmentation Pipeline
# --------------------------

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --------------------------
# Load ISIC2019 Dataset
# --------------------------

def load_isic_dataset(base_dir):
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for category, idx in CATEGORIES.items():
        train_dir = os.path.join(base_dir, "Train", category)
        val_dir = os.path.join(base_dir, "Validation", category)

        # Load train images
        train_imgs = Dataset_loader(train_dir, RESIZE=IMAGE_SIZE)
        train_labels = np.full(len(train_imgs), idx)
        X_train.extend(train_imgs)
        Y_train.extend(train_labels)

        # Load test images
        test_imgs = Dataset_loader(val_dir, RESIZE=IMAGE_SIZE)
        test_labels = np.full(len(test_imgs), idx)
        X_test.extend(test_imgs)
        Y_test.extend(test_labels)

    # Convert to numpy arrays
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    # Shuffle
    indices = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[indices], Y_train[indices]
    indices = np.random.permutation(len(X_test))
    X_test, Y_test = X_test[indices], Y_test[indices]

    # Normalize and convert to tensor
    X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    X_test = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    return X_train, Y_train, X_test, Y_test

# --------------------------
# Non-IID Data Partitioning
# --------------------------

def dirichlet_split_dataset(X, y, num_clients, alpha=0.5):
    labels = y.numpy()
    unique_labels = np.unique(labels)
    client_datasets = [[] for _ in range(num_clients)]
    class_indices = [np.where(labels == c)[0] for c in unique_labels]

    for c_idx, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        cumulative = 0
        for i in range(num_clients):
            split_size = int(proportions[i] * len(indices))
            client_datasets[i].extend(indices[cumulative:cumulative+split_size])
            cumulative += split_size

    return [
        TensorDataset(X[idxs], y[idxs]) for idxs in client_datasets
    ]

# --------------------------
# Label Smoothing Loss
# --------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, num_classes=8):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        targets = targets.view(-1).long()
        log_preds = F.log_softmax(inputs, dim=-1)
        nll_loss = F.nll_loss(log_preds, targets, reduction='none')
        smooth_loss = -log_preds.mean(dim=-1)
        loss = (1 - self.alpha) * nll_loss + self.alpha * smooth_loss
        return loss.mean()

# --------------------------
# Custom CNN Model
# --------------------------

class CNNModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CNNModel, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        features = list(alexnet.features.children())
        self.layer1 = nn.Sequential(*features[0:2])  # Conv1 + ReLU
        self.pool1 = features[2]  # MaxPool
        self.layer2 = nn.Sequential(*features[3:5])  # Conv2 + ReLU
        self.pool2 = features[5]  # MaxPool

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --------------------------
# Training & Evaluation
# --------------------------

def evaluate_model(model, val_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=CATEGORIES.keys()))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES.keys(), yticklabels=CATEGORIES.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    scaler = GradScaler()
    model.train()
    for epoch in range(epochs):
        running_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    evaluate_model(model, val_loader)

# --------------------------
# Plotting Function
# --------------------------

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()

# --------------------------
# Main FL Function
# --------------------------

def main():
    base_dir = "/content/drive/MyDrive/data"
    X_train, Y_train, X_test, Y_test = load_isic_dataset(base_dir)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Split into non-IID clients
    client_datasets = dirichlet_split_dataset(X_train, Y_train, CLIENTS, DIRICHLET_ALPHA)
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]

    # Initialize global and client models
    global_model = CNNModel(NUM_CLASSES).to(DEVICE)
    client_models = [CNNModel(NUM_CLASSES).to(DEVICE) for _ in range(CLIENTS)]

    # Training Loop
    for round in range(ROUNDS):
        print(f"\n=== Federated Round {round+1}/{ROUNDS} ===")
        global_weights = global_model.state_dict()

        updated_models = []
        for i, model in enumerate(client_models):
            model.load_state_dict(global_weights)
            criterion = LabelSmoothingCrossEntropy(alpha=0.1, num_classes=NUM_CLASSES)
            optimizer = Adam(model.parameters(), lr=1e-4)
            print(f"Training Client {i+1}")
            train_model(model, client_loaders[i], test_loader, criterion, optimizer, epochs=LOCAL_EPOCHS)
            updated_models.append(model)

        # Aggregate updates
        avg_weights = {}
        for key in global_weights.keys():
            tensors = [model.state_dict()[key] for model in updated_models]
            avg_tensor = torch.mean(torch.stack(tensors), dim=0)
            avg_weights[key] = avg_tensor

        global_model.load_state_dict(avg_weights)
        client_models = [CNNModel(NUM_CLASSES).to(DEVICE) for _ in range(CLIENTS)]
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

    # Final evaluation
    evaluate_model(global_model, test_loader)
    torch.save(global_model.state_dict(), 'final_isic_fedmedtl_model.pth')

if __name__ == "__main__":
    main()
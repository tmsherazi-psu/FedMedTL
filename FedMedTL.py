# Import Required Libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.models as models
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import random
from torch.cuda.amp import autocast, GradScaler

# --- Custom Label Smoothing Loss ---
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, num_classes=8):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        targets = targets.argmax(dim=1)
        log_probs = self.log_softmax(inputs)
        nll_loss = F.nll_loss(log_probs, targets, reduction='mean')
        smooth_loss = -log_probs.mean(dim=1).mean()
        return (1 - self.alpha) * nll_loss + self.alpha * smooth_loss


# --- CNN Model Architecture (AlexNet with Knowledge Distillation Support) ---
class CNNModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CNNModel, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        features = list(alexnet.features.children())
        self.layer1 = nn.Sequential(*features[0:2])
        self.pool1 = features[2]
        self.layer2 = nn.Sequential(*features[3:5])
        self.pool2 = features[5]

        # Fully connected layers
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


# --- Dirichlet Distribution to Simulate Non-IID Data ---
def dirichlet_distribution(alpha, num_clients, num_classes, num_samples_per_class):
    """
    Generate Dirichlet distributed data to simulate non-IID data distribution among clients
    """
    total_samples = num_samples_per_class * num_classes
    class_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)
    client_labels = [np.concatenate([np.full(int(p * num_samples_per_class), cls) for cls, p in enumerate(probs)])
                      for probs in class_distribution]
    return client_labels


# --- Dataset Loading & Transformation ---
def load_data():
    base_dir = "/content/drive/MyDrive/data"
    # Define category labels and directories
    categories = {
        'AK': 0,
        'BCC': 1,
        'BKL': 2,
        'DF': 3,
        'MEL': 4,
        'NV': 5,
        'SCC': 6,
        'VASC': 7
    }

    def load_images_from_folder(folder):
        images = []
        for filename in tqdm(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            if img_path.endswith(".png"):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
        return np.array(images)

    # Load images for each category
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for cat, label in categories.items():
        train_dir = os.path.join(base_dir, f"Train/{cat}")
        test_dir = os.path.join(base_dir, f"Validation/{cat}")

        train_imgs = load_images_from_folder(train_dir)
        test_imgs = load_images_from_folder(test_dir)

        train_labels = np.full(len(train_imgs), label)
        test_labels = np.full(len(test_imgs), label)

        X_train.extend(train_imgs)
        Y_train.extend(train_labels)
        X_test.extend(test_imgs)
        Y_test.extend(test_labels)

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    # Shuffle
    idxs = np.arange(len(X_train))
    np.random.shuffle(idxs)
    X_train, Y_train = X_train[idxs], Y_train[idxs]

    idxs = np.arange(len(X_test))
    np.random.shuffle(idxs)
    X_test, Y_test = X_test[idxs], Y_test[idxs]

    # Convert to tensors
    X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    X_test = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader


# --- Evaluate the Model Function ---
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# --- Training Function with Knowledge Distillation Option ---
def train_model(model, train_loader, val_loader, criterion, optimizer, teacher_model=None, distill_alpha=0.5, num_epochs=50):
    scaler = GradScaler()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Knowledge Distillation
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(images)
                    soft_loss = F.kl_div(F.log_softmax(outputs / 3.0, dim=1),
                                         F.softmax(teacher_outputs / 3.0, dim=1),
                                         reduction='batchmean') * (3.0 ** 2)
                    loss = (1 - distill_alpha) * loss + distill_alpha * soft_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies


# --- Main Execution ---
def main():
    train_loader, val_loader = load_data()

    # Initialize global model and client models
    global_model = CNNModel(num_classes=8)
    client_models = [CNNModel(num_classes=8) for _ in range(5)]  # Assume 5 clients

    # Federated Learning Setup
    num_rounds = 10
    local_epochs = 5
    learning_rate = 7e-5

    for round in range(num_rounds):
        print(f"\n=== Federated Round {round+1}/{num_rounds} ===")
        global_weights = global_model.state_dict()

        # Client training
        updated_models = []
        for i, model in enumerate(client_models):
            model.load_state_dict(global_weights)
            criterion = LabelSmoothingCrossEntropy(alpha=0.1)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print(f"Training Client {i+1}")
            train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=local_epochs)
            updated_models.append(model)

        # Dynamic Clustering (simplified example)
        logits_list = [model(next(iter(train_loader))[0]) for model in updated_models]
        cluster_ids = dynamic_clustering(logits_list)

        # Homomorphic Encryption (placeholder)
        encrypted_weights = [encrypt_weights(model.state_dict()) for model in updated_models]

        # Aggregate per-cluster weights
        new_global_weights = aggregate_cluster_weights(encrypted_weights, cluster_ids)

        # Update global model
        global_model.load_state_dict(new_global_weights)

        # Set updated global model to all clients
        client_models = [CNNModel(num_classes=8) for _ in range(5)]
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

    # Final Evaluation
    evaluate_model(global_model, val_loader)
    torch.save(global_model.state_dict(), 'final_trained_model.pth')


# --- Placeholder Functions for FL Components ---

def dynamic_clustering(logits_list):
    """Placeholder for DBSCAN-based clustering."""
    return [0, 1, 0, 1, 2]  # Dummy clusters


def encrypt_weights(weights):
    """Placeholder for homomorphic encryption."""
    return weights


def aggregate_cluster_weights(weights_list, cluster_ids):
    """Aggregate weights within each cluster."""
    avg_weights = {}
    for key in weights_list[0].keys():
        tensors = [w[key] for w in weights_list]
        avg_tensor = torch.mean(torch.stack(tensors), dim=0)
        avg_weights[key] = avg_tensor
    return avg_weights


if __name__ == "__main__":
    main()
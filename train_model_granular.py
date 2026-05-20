import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations with augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset from granular classification folder
data_dir = "/Users/manasimaxin/Downloads/caries detection/granular_classification"
print(f"Loading dataset from: {data_dir}")

full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
print(f"Classes: {full_dataset.classes}")
print(f"Total images: {len(full_dataset)}")

# Calculate class weights to handle imbalance (Rampant_Caries is 68%)
class_indices = np.array([y for x, y in full_dataset.samples])
class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"Class weights: {class_weights}")

# Split into train and val (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Train samples: {train_size}, Val samples: {val_size}")

# Model setup
num_classes = len(full_dataset.classes)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Loss and optimizer with weighted loss for imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    # Training loop
    num_epochs = 15
    print(f"\nTraining for {num_epochs} epochs with weighted loss...")
    print("=" * 70)

    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / train_size
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = val_correct / val_total
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        print(f"Epoch {epoch+1:2d}: Loss {epoch_loss:.4f}, Train Acc {epoch_acc:.4f}, Val Acc {val_acc:.4f}")

    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Save the model
    torch.save(model.state_dict(), "teeth_model_granular.pth")
    with open("classes_granular.txt", "w") as f:
        f.write("\n".join(full_dataset.classes))
    print("\nModel saved to teeth_model_granular.pth")
    print("Classes saved to classes_granular.txt")

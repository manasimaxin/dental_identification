import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# New 5 Categories
NEW_CLASSES = [
    "early stage lesions",
    "cavitated lesions",
    "fissure caries",
    "smooth surface caries",
    "dental root caries"
]

# Mapping from original 8 to new 5
MAPPING = {
    "Acute_Caries": "early stage lesions",
    "Primary_Caries": "early stage lesions",
    "Chronic_Caries": "cavitated lesions",
    "Rampant_Caries": "cavitated lesions",
    "Secondary_Caries": "cavitated lesions",
    "Pit_and_Fissure_Caries": "fissure caries",
    "Smooth_Surface_Caries": "smooth surface caries",
    "Root_Caries": "dental root caries"
}

class MappedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.original_dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.classes = NEW_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for path, old_idx in self.original_dataset.samples:
            old_class = self.original_dataset.classes[old_idx]
            if old_class in MAPPING:
                new_class = MAPPING[old_class]
                new_idx = self.class_to_idx[new_class]
                self.samples.append((path, new_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    data_dir = "/Users/manasimaxin/Downloads/caries detection/granular_classification"
    
    # Advanced augmentation for high accuracy
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = MappedDataset(data_dir, transform=train_transforms)
    print(f"Total samples after mapping: {len(full_dataset)}")
    
    # Calculate weights for imbalance
    labels = [s[1] for s in full_dataset.samples]
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Override transform for val set
    val_dataset.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 5)
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    
    num_epochs = 12
    best_acc = 0
    
    print("Starting training...")
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
            
        train_acc = correct / total
        
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
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "teeth_model_5_classes.pth")
            print(f"New best model saved with accuracy: {best_acc:.4f}")
            
    with open("classes_5.txt", "w") as f:
        f.write("\n".join(NEW_CLASSES))
        
    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()

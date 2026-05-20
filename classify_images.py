import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Settings
model_path = "teeth_model.pth"
classes_path = "classes.txt"
unlabeled_dir = Path("/Users/manasimaxin/Downloads/INTRA ORAL IMAGES FROM KAGGLE/INTRA ORAL IMAGES KAGGLE")
output_dir = Path("/Users/manasimaxin/teeth_classified")

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def classify():
    # Load classes
    if not os.path.exists(classes_path):
        print("Error: classes.txt not found. Run training first.")
        return
    with open(classes_path, "r") as f:
        classes = f.read().splitlines()
    
    # Model setup
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    
    if not os.path.exists(model_path):
        print("Error: teeth_model.pth not found. Wait for training to finish.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Process images
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    for c in classes:
        (output_dir / c).mkdir(exist_ok=True)
        
    images = list(unlabeled_dir.glob("*.jpg")) + list(unlabeled_dir.glob("*.jpeg")) + list(unlabeled_dir.glob("*.png"))
    print(f"Classifying {len(images)} images from {unlabeled_dir}...")
    
    with torch.no_grad():
        for img_path in tqdm(images):
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                
                outputs = model(input_tensor)
                _, predicted = outputs.max(1)
                class_name = classes[predicted.item()]
                
                # Copy image to classified folder
                shutil.copy(img_path, output_dir / class_name / img_path.name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    classify()
    print(f"Classification complete. Results in {output_dir}")

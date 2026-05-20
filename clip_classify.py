import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import shutil
from tqdm import tqdm

# Configuration
unlabeled_dirs = [
    Path("/Users/manasimaxin/Downloads/INTRA ORAL IMAGES FROM KAGGLE/INTRA ORAL IMAGES KAGGLE"),
    Path("/Users/manasimaxin/Downloads/INTRA ORAL IMAGES FROM KAGGLE/tooth")
]
output_dir = Path("/Users/manasimaxin/Downloads/caries detection/granular_classification")

# Define Clinical Labels and Descriptive Prompts
class_descriptions = {
    "Pit_and_Fissure_Caries": "An intraoral photo showing dental decay in the pits and fissures on the biting surfaces of molars.",
    "Smooth_Surface_Caries": "An intraoral photo showing dental decay on the smooth flat surfaces of the teeth.",
    "Root_Caries": "An intraoral photo showing dental decay on the exposed root surface of a tooth near the gums.",
    "Acute_Caries": "An intraoral photo showing rapidly progressing, soft, light-colored dental decay.",
    "Chronic_Caries": "An intraoral photo showing slow-progressing, hard, dark-colored or leathery dental decay.",
    "Rampant_Caries": "An intraoral photo showing widespread and severe dental decay affecting multiple teeth at once.",
    "Primary_Caries": "An intraoral photo showing a new cavity on a tooth surface that has never been filled.",
    "Secondary_Caries": "An intraoral photo showing dental decay at the edges of an existing dental filling or crown."
}

labels = list(class_descriptions.keys())
prompts = [class_descriptions[l] for l in labels]

def run_zero_shot():
    # Setup Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Prepare Output Dirs
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    for label in labels:
        (output_dir / label).mkdir(exist_ok=True)

    # Gather Images Recursively
    image_paths = []
    for d in unlabeled_dirs:
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(list(d.rglob(ext)))

    print(f"Starting full classification of {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Prepare inputs
            inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get probabilities
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get best label
            idx = probs.argmax().item()
            best_label = labels[idx]

            # Copy to target folder
            # Prefix with parent name to avoid collision in recursive search
            dest_name = f"{img_path.parent.name}_{img_path.name}"
            dest = output_dir / best_label / dest_name
            shutil.copy(img_path, dest)
            
        except Exception as e:
            pass # Skip errors for the full run to maintain speed

    print(f"\nClassification complete. Results in: {output_dir}")

if __name__ == "__main__":
    run_zero_shot()

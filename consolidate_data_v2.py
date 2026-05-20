"""
Consolidate training data from INTRA ORAL IMAGES FROM KAGGLE folder.
Maps scattered sources to class folders.
"""
from pathlib import Path
import shutil
import os
from tqdm import tqdm

# Output consolidated dataset
output_dir = Path("/Users/manasimaxin/teeth_dataset_consolidated_v2")
output_dir.mkdir(exist_ok=True)

# Define all source-to-class mappings
SOURCES = {
    "Caries": [
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset1/archive/teeth_dataset/teeth_dataset/Trianing/caries",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset1/archive/teeth_dataset/teeth_dataset/test/caries",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset1/caries",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset2/archive/teeth_dataset/test/caries",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset2/archive/teeth_dataset/train/caries",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset2/caries",
        "INTRA ORAL IMAGES FROM KAGGLE/SET 2 /train/cavity",
        "INTRA ORAL IMAGES FROM KAGGLE/SET 2 /test/cavity",
    ],
    "Healthy": [
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset1/archive/teeth_dataset/teeth_dataset/Trianing/without_caries",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset1/archive/teeth_dataset/teeth_dataset/test/no-caries",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset2/archive/teeth_dataset/test/no-caries",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset2/archive/teeth_dataset/train/no-caries",
        "INTRA ORAL IMAGES FROM KAGGLE/SET 2 /train/no_cavity",
        "INTRA ORAL IMAGES FROM KAGGLE/SET 2 /test/no_cavity",
    ],
    "Gingivitis": [
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset4/archive/Gingivitis/Gingivitis",
    ],
    "Hypodontia": [
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset4/archive/hypodontia/hypodontia",
    ],
    "Calculus": [
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset4/archive/Calculus/Calculus",
    ],
    "Ulcer": [
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset4/archive/Mouth Ulcer/Mouth Ulcer/ulcer original dataset/ulcer original dataset",
    ],
    "Discoloration": [
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset4/archive/Tooth Discoloration/Tooth Discoloration/tooth discoloration original dataset/tooth discoloration original dataset",
    ],
    "Xray": [
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/x-ray/images",
        "INTRA ORAL IMAGES FROM KAGGLE/tooth/dataset3/Dental Cavity Dataset/Dataset/x-ray/images",
    ],
}

total_copied = 0

for class_name, sources in SOURCES.items():
    class_dir = output_dir / class_name
    class_dir.mkdir(exist_ok=True)
    class_count = 0
    
    for src_rel in sources:
        src_path = Path(src_rel)
        if not src_path.exists():
            print(f"  Warning: {src_rel} not found")
            continue
        
        # Find all images
        images = list(src_path.glob("*.jpg")) + list(src_path.glob("*.jpeg")) + list(src_path.glob("*.png"))
        
        for img in tqdm(images, desc=f"{class_name} ({src_rel.split('/')[-1]})"):
            try:
                dst = class_dir / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
                    class_count += 1
            except Exception as e:
                print(f"Error copying {img}: {e}")
    
    print(f"{class_name}: {class_count} images")
    total_copied += class_count

print(f"\nTotal images consolidated: {total_copied}")
print(f"Consolidated dataset saved to: {output_dir}")

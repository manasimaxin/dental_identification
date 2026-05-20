import os
import shutil
from pathlib import Path

# Define base paths
base_path = Path("/Users/manasimaxin/Downloads/INTRA ORAL IMAGES FROM KAGGLE/")
target_path = Path("/Users/manasimaxin/teeth_dataset_consolidated")

# Define class mapping (Source Dir -> Class Name)
mapping = {
    # Caries
    base_path / "tooth/dataset1/caries": "Caries",
    base_path / "tooth/dataset1/archive/teeth_dataset/teeth_dataset/Trianing/caries": "Caries",
    base_path / "SET 2 /train/cavity": "Caries",
    base_path / "SET 2 /test/cavity": "Caries",
    base_path / "tooth/dataset4/archive/Data caries/Data caries/caries orignal data set/done": "Caries",
    base_path / "tooth/dataset4/archive/Data caries/Data caries/caries augmented data set/preview": "Caries",
    
    # Healthy / No Cavity
    base_path / "tooth/dataset1/archive/teeth_dataset/teeth_dataset/Trianing/without_caries": "Healthy",
    base_path / "SET 2 /train/no_cavity": "Healthy",
    base_path / "SET 2 /test/no_cavity": "Healthy",
    
    # Gingivitis
    base_path / "tooth/dataset4/archive/Gingivitis/Gingivitis": "Gingivitis",
    
    # Calculus
    base_path / "tooth/dataset4/archive/Calculus/Calculus": "Calculus",
    
    # Discoloration
    base_path / "tooth/dataset4/archive/Tooth Discoloration/Tooth Discoloration/tooth discoloration original dataset/tooth discoloration original dataset": "Discoloration",
    base_path / "tooth/dataset4/archive/Tooth Discoloration/Tooth Discoloration/Tooth_discoloration_augmented_dataser/preview": "Discoloration",
    
    # Ulcer
    base_path / "tooth/dataset4/archive/Mouth Ulcer/Mouth Ulcer/ulcer original dataset/ulcer original dataset": "Ulcer",
    base_path / "tooth/dataset4/archive/Mouth Ulcer/Mouth Ulcer/Mouth_Ulcer_augmented_DataSet/preview": "Ulcer",
    
    # Hypodontia
    base_path / "tooth/dataset4/archive/hypodontia/hypodontia": "Hypodontia",
    
    # X-ray
    base_path / "tooth/x-ray/images": "Xray",
    base_path / "tooth/dataset3/Dental Cavity Dataset/Dataset/x-ray/images": "Xray",
}

def consolidate():
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    
    for src_dir, class_name in mapping.items():
        if not src_dir.exists():
            print(f"Warning: Source directory {src_dir} does not exist.")
            continue
            
        class_dir = target_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        print(f"Processing {src_dir} -> {class_name}...")
        for img_path in src_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                # To avoid name collisions, prefix with a hash of the source dir path
                prefix = str(hash(str(src_dir)))[:8]
                dest_name = f"{prefix}_{img_path.name}"
                dest_path = class_dir / dest_name
                
                # Symlink to save space
                try:
                    os.symlink(img_path, dest_path)
                except FileExistsError:
                    pass

if __name__ == "__main__":
    consolidate()
    print("Consolidation complete.")

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import json

# Configuration
input_dir = Path("/Users/manasimaxin/Downloads/caries detection/granular_classification")
output_dir = Path("/Users/manasimaxin/Downloads/caries detection/final caries labelled")
threshold = 0.15  # Detection confidence threshold

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

def detect_and_annotate():
    print(f"Loading Owlv2 model on {device}...")
    model_id = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    # Texts to detect
    texts = [["dental caries", "tooth cavity", "dental decay"]]

    # Find all images in the classified folders
    image_paths = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.jpeg")) + list(input_dir.rglob("*.png"))
    
    print(f"Detecting caries in {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)

            # Target image sizes (height, width) to rescale box predictions
            target_sizes = torch.Tensor([image.size[::-1]]).to(device)
            # Use grounded object detection post-processing and supply text labels
            results = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=texts
            )

            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            detections = results[0]
            img_h, img_w = img_cv.shape[:2]
            image_results = []
            for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
                box = [float(i) for i in box.tolist()]
                x1, y1, x2, y2 = map(int, box)
                # compute area ratio as a proxy for severity
                box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                img_area = max(1, img_w * img_h)
                area_ratio = box_area / img_area

                # severity rules (simple heuristic): combine area ratio and confidence
                if score < 0.25:
                    severity = "very_low_confidence"
                else:
                    if area_ratio < 0.002:
                        severity = "mild"
                    elif area_ratio < 0.01:
                        severity = "moderate"
                    else:
                        severity = "severe"

                # draw semi-transparent mask for the detected area
                overlay = img_cv.copy()
                color = (0, 0, 255) if severity in ("moderate", "severe") else (0, 165, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                alpha = 0.35
                cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0, img_cv)

                # draw bounding box and label text
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 150), 2)
                label_text = f"{severity} ({score:.2f})"
                cv2.putText(img_cv, label_text, (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                image_results.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": float(score),
                    "severity": severity,
                    "area_ratio": float(area_ratio),
                })

            # Save annotated image maintaining folder structure
            relative_path = img_path.relative_to(input_dir)
            save_path = output_dir / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img_cv)

            # Save per-image JSON summary
            json_path = save_path.with_suffix('.json')
            with open(json_path, 'w') as jf:
                json.dump({
                    "image": str(img_path),
                    "detections": image_results
                }, jf, indent=2)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print(f"\nDetection complete. Annotated images saved in: {output_dir}")

if __name__ == "__main__":
    detect_and_annotate()

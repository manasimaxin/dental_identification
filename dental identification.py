import io
import os
import base64
import json
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

from transformers import Owlv2Processor, Owlv2ForObjectDetection

app = Flask(__name__, static_folder='static')

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading models on {device}...")

# Load Owlv2 for detection
model_id = "google/owlv2-base-patch16-ensemble"
processor = Owlv2Processor.from_pretrained(model_id)
owlv2_model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
owlv2_model.eval()

# Load granular classification model
print("Loading granular classification model...")
classifier = resnet18(weights=None)
classifier.fc = torch.nn.Linear(512, 8)
state_dict = torch.load('teeth_model_granular.pth', map_location=device)
classifier.load_state_dict(state_dict)
classifier.to(device)
classifier.eval()

# Load class labels
with open('classes_granular.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Classification transforms
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

THRESHOLD = 0.15

def classify_image_patch(pil_patch):
    """Classify a single image patch using granular model."""
    img_tensor = classify_transform(pil_patch).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
    return classes[pred_idx], confidence

def analyze_image(pil_image):
    image = pil_image.convert('RGB')
    inputs = processor(text=[["dental caries", "tooth cavity", "dental decay"]], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = owlv2_model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    texts = [["dental caries", "tooth cavity", "dental decay"]]
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=THRESHOLD, text_labels=texts
    )
    detections = results[0]

    # Convert to OpenCV image
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_h, img_w = img_cv.shape[:2]
    image_results = []
    
    # If no detections, still try classification on full image
    if len(detections["scores"]) == 0:
        severity_class, confidence = classify_image_patch(image)
        return None, [{
            "type": "full_image_classification",
            "severity_class": severity_class,
            "confidence": float(confidence),
            "message": "No specific cavities detected; full image severity classification"
        }]
    
    for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        box = [float(i) for i in box.tolist()]
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
        
        # Classify the detected region
        detected_patch = image.crop((x1, y1, x2, y2))
        severity_class, class_confidence = classify_image_patch(detected_patch)
        
        box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        img_area = max(1, img_w * img_h)
        area_ratio = box_area / img_area

        # Color based on severity class
        severity_colors = {
            "Acute_Caries": (255, 100, 0),
            "Chronic_Caries": (255, 0, 0),
            "Pit_and_Fissure_Caries": (200, 0, 150),
            "Primary_Caries": (100, 0, 255),
            "Rampant_Caries": (0, 0, 255),
            "Root_Caries": (255, 0, 255),
            "Secondary_Caries": (0, 165, 255),
            "Smooth_Surface_Caries": (0, 255, 255)
        }
        color = severity_colors.get(severity_class, (100, 100, 255))

        # Overlay
        overlay = img_cv.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0, img_cv)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{severity_class} ({class_confidence:.1%})"
        cv2.putText(img_cv, label_text, (x1, max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        image_results.append({
            "bbox": [x1, y1, x2, y2],
            "detection_score": float(score),
            "severity_class": severity_class,
            "classification_confidence": float(class_confidence),
            "area_ratio": float(area_ratio),
        })

    # Encode annotated image as PNG base64
    _, png = cv2.imencode('.png', img_cv)
    b64 = base64.b64encode(png.tobytes()).decode('utf-8')
    data_url = f"data:image/png;base64,{b64}"

    return data_url, image_results if image_results else [{"message": "No detections"}]


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "no image provided"}), 400
    f = request.files['image']
    try:
        pil_image = Image.open(f.stream)
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    data_url, detections = analyze_image(pil_image)
    return jsonify({"image": data_url, "detections": detections})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

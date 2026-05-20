import io
import os
import base64
import json
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18

app = Flask(__name__, static_folder='static')

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading models on {device}...")

# Load 5-class classification model
print("Loading 5-category classification model...")
classifier = resnet18(weights=None)
classifier.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, 5)
)
state_dict = torch.load('teeth_model_5_classes.pth', map_location=device)
classifier.load_state_dict(state_dict)
classifier.to(device)
classifier.eval()

# Load new class labels
with open('classes_5.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Classification transforms
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def check_image_quality(cv_img):
    """Checks sharpness and lighting."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Lighting (Mean brightness)
    brightness = np.mean(gray)
    
    reasons = []
    if sharpness < 100:
        reasons.append("Image is too blurry. Please hold the camera steady.")
    if brightness < 40:
        reasons.append("Lighting is too dark. Please use a flash or move to a well-lit area.")
    if brightness > 220:
        reasons.append("Image is overexposed. Avoid direct bright light.")
        
    return len(reasons) == 0, reasons

def analyze_image(pil_image, patient_info=None):
    # Resize to model input size directly for speed
    image = pil_image.resize((224, 224)).convert('RGB')
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Quality Check (on resized image)
    is_quality_ok, quality_reasons = check_image_quality(img_cv)
    if not is_quality_ok:
        return None, {"error": "Quality check failed", "reasons": quality_reasons}

    # 2. Fast Classification
    img_tensor = classify_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
    
    # Create simple result
    action_needed = "Urgent Treatment Required" if pred_idx in [1, 4] else "Treatment Recommended"
    advice = "Professional cleaning and restoration likely needed." if pred_idx > 0 else "Continue regular brushing."

    return img_cv, [{
        "bbox": [0, 0, 224, 224],
        "caries_type": classes[pred_idx],
        "confidence": f"{confidence:.1%}",
        "action_needed": action_needed,
        "summary": f"Detected {classes[pred_idx]} with {confidence:.1%} confidence.",
        "advice": advice
    }]


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    f = request.files['image']
    patient_data = request.form.get('patient_data')
    if patient_data:
        patient_info = json.loads(patient_data)
    else:
        patient_info = {"id": str(uuid.uuid4())[:8]}

    try:
        pil_image = Image.open(f.stream)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    result_img, analysis = analyze_image(pil_image, patient_info)
    
    if result_img is None:
        return jsonify(analysis), 422 # Quality/Visibility errors

    # Encode result image
    _, png = cv2.imencode('.png', result_img)
    b64 = base64.b64encode(png.tobytes()).decode('utf-8')
    data_url = f"data:image/png;base64,{b64}"

    return jsonify({
        "patient_id": patient_info.get('id'),
        "image": data_url,
        "results": analysis
    })


if __name__ == '__main__':
    # Hugging Face Spaces requires port 7860
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)

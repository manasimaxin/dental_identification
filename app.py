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

# Grad-CAM components
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def save_activation(module, input, output):
            self.activations = output
            
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        return heatmap

grad_cam = GradCAM(classifier, classifier.layer4)

THRESHOLD = 0.15

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
    image = pil_image.convert('RGB')
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Quality Checks
    is_quality_ok, quality_reasons = check_image_quality(img_cv)
    if not is_quality_ok:
        return None, {"error": "Quality check failed", "reasons": quality_reasons}

    # 2. Visibility Check & Detection
    # Detect both "tooth" and "caries" for visibility verification
    prompt_texts = [["tooth", "dental caries", "tooth cavity"]]
    inputs = processor(text=prompt_texts, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = owlv2_model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=THRESHOLD, text_labels=prompt_texts
    )
    detections = results[0]
    
    # Check if a tooth is actually visible
    tooth_detected = any(label == "tooth" for label in detections["labels"])
    if not tooth_detected:
         return None, {"error": "Visibility check failed", "reasons": ["No tooth clearly visible in the image."]}

    # Filter only for caries detections for analysis
    caries_indices = [i for i, label in enumerate(detections["labels"]) if label != "tooth"]
    
    img_h, img_w = img_cv.shape[:2]
    image_results = []
    
    # If no specific caries detected, classify full image
    if len(caries_indices) == 0:
        # Just run classification on the main tooth area (or full image)
        img_tensor = classify_transform(image).unsqueeze(0).to(device)
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        
        # Heatmap for full image
        heatmap = grad_cam.generate_heatmap(img_tensor, pred_idx)
        heatmap = cv2.resize(heatmap, (img_w, img_h))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay
        result_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
        
        return result_img, [{
            "type": "General Classification",
            "caries_type": classes[pred_idx],
            "confidence": f"{probs[pred_idx].item():.1%}",
            "action_needed": "Consult dentist for checkup" if pred_idx > 0 else "Routine checkup",
            "summary": f"No specific cavity detected, but general tooth surface shows signs of {classes[pred_idx]}.",
            "advice": "Maintain good oral hygiene and visit a dentist annually."
        }]

    # Process each detected caries area
    for idx in caries_indices:
        score = detections["scores"][idx]
        box = detections["boxes"][idx]
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
        
        patch = image.crop((x1, y1, x2, y2))
        img_tensor = classify_transform(patch).unsqueeze(0).to(device)
        
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        
        # Generate Heatmap for this patch
        heatmap = grad_cam.generate_heatmap(img_tensor, pred_idx)
        heatmap = cv2.resize(heatmap, (x2-x1, y2-y1))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original image patch
        img_cv[y1:y2, x1:x2] = cv2.addWeighted(img_cv[y1:y2, x1:x2], 0.5, heatmap_color, 0.5, 0)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 255, 255), 2)

        action_needed = "Urgent Treatment Required" if pred_idx in [1, 4] else "Treatment Recommended"
        advice = "Professional cleaning and restoration likely needed." if pred_idx > 0 else "Continue regular brushing."

        image_results.append({
            "bbox": [x1, y1, x2, y2],
            "caries_type": classes[pred_idx],
            "confidence": f"{confidence:.1%}",
            "action_needed": action_needed,
            "summary": f"Detected {classes[pred_idx]} with {confidence:.1%} confidence.",
            "advice": advice
        })

    return img_cv, image_results


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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

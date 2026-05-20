# Dental Diagnosis AI

An AI-powered web application for early detection and severity classification of dental caries using deep learning.

## Features
- **Smart Patient Registration**: Integrated patient data tracking with unique IDs.
- **Auto-Capture Camera**: Quality-controlled imaging (sharpness, lighting, and tooth visibility verification) to ensure professional-grade inputs.
- **Caries Identification**: Uses Google's `Owlv2` for object detection and a custom `ResNet18` model for granular severity classification.
- **Visualization**: Generates real-time **Grad-CAM heatmaps** to highlight detected caries.
- **Detailed Reporting**: Provides actionable diagnosis, severity statistics, and clinical advice.

## Installation & Running Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dental-ai-diagnosis.git
   cd dental-ai-diagnosis
   ```

2. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   python app.py
   ```
   *The app will be accessible at `http://localhost:5000`.*

## Deployment
To deploy this as a public website accessible on your mobile devices:
1. Push this code to a public GitHub repository.
2. Create a new Space on [Hugging Face](https://huggingface.co/spaces/new).
3. Select **Flask** as the SDK and link your repository.
4. The space will build automatically and provide a public URL for your application.

---
*Built with PyTorch, Flask, and Transformers.*

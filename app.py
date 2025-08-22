from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import yaml
import os
import traceback

# This is the key change: Import the YOLO class directly
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Your trained model is expected to be in the same directory as app.py
MODEL_PATH = 'best.pt'
IMG_SIZE = 416

# Try to load the model
try:
    print("Attempting to load model using ultralytics.YOLO...")
    # Use the YOLO class to load your custom model. This is the official method.
    model = YOLO(MODEL_PATH)
    model.conf = 0.25
    model.iou = 0.45
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    model = None
    print("Running without model - will show demo message")

# Load class names from data.yaml
try:
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
        CLASS_NAMES = data_config['names']
    print(f"Classes loaded: {CLASS_NAMES}")
except Exception as e:
    print(f"Error loading data.yaml: {e}")
    CLASS_NAMES = ['mug', 'spoon']  # Fallback
    print("Using fallback class names")

def process_detection(image, results):
    try:
        if not results or not results[0].boxes:
            return image, "No objects detected", []

        detections_data = results[0].boxes.data.cpu().numpy()
        
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        detected_objects = []
        for det in detections_data:
            x1, y1, x2, y2, confidence, class_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_name = CLASS_NAMES[int(class_id)]
            
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            label = f"{class_name}: {confidence:.2f}"
            text_x = x1
            text_y = max(y1 - 25, 0)
            draw.text((text_x, text_y), label, fill='red', font=font)
            
            detected_objects.append({
                'class': class_name,
                'confidence': f"{confidence:.2%}",
                'bbox': [x1, y1, x2, y2]
            })
        
        object_counts = {}
        for obj in detected_objects:
            object_counts[obj['class']] = object_counts.get(obj['class'], 0) + 1
        
        summary = [f"{count} {cls}{'s' if count > 1 else ''}" for cls, count in object_counts.items()]
        summary_text = f"Detected: {', '.join(summary)}" if summary else "No objects detected"
        
        return image, summary_text, detected_objects
    except Exception as e:
        return image, f"Error processing detection: {str(e)}", []

def predict_image(image):
    try:
        if model is None:
            return None, "Model not loaded", []
        
        # Run inference using the YOLO model. The imgsz argument is for resizing.
        results = model(image, imgsz=IMG_SIZE)
        
        annotated_image, summary, detections = process_detection(image.copy(), results)
        
        return annotated_image, summary, detections
    except Exception as e:
        traceback.print_exc()
        return None, f"Error during prediction: {str(e)}", []

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file'] if 'file' in request.files else None
        if not file or file.filename == '':
            return jsonify({'error': 'No file uploaded'})

        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        annotated_image, summary, detections = predict_image(image)
        
        if annotated_image is None:
            return jsonify({'error': summary})
        
        return jsonify({
            'summary': summary,
            'detections': detections,
            'annotated_image': pil_to_base64(annotated_image),
            'total_objects': len(detections)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
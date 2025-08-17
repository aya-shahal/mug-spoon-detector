from flask import Flask, request, render_template, jsonify
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import yaml
import os
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load YOLOv5 model
MODEL_PATH = 'best.pt'  # Your trained model
IMG_SIZE = 416  # Same size you used for training

# Try to load the model
try:
    import pathlib
    # Fix for Windows path issues
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
    # Load the model (YOLOv5 format)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    model.conf = 0.25  # confidence threshold
    model.iou = 0.45   # IoU threshold
    print("Model loaded successfully!")
    
    # Restore original pathlib
    pathlib.PosixPath = temp
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    print("Running without model - will show demo message")

# Load class names from data.yaml
try:
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
        CLASS_NAMES = data_config['names']
    print(f"Classes loaded: {CLASS_NAMES}")
except:
    CLASS_NAMES = {0: 'mug', 1: 'spoon'}  # fallback
    print("Using fallback class names")

def process_detection(image, results):
    """
    Process YOLOv5 detection results and draw bounding boxes
    """
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Get detection results
        detections = results.pandas().xyxy[0]  # get detections as pandas dataframe
        
        if len(detections) == 0:
            return image, "No objects detected", []
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        
        # Try to use a better font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        detected_objects = []
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1-25), label, fill='red', font=font)
            
            detected_objects.append({
                'class': class_name,
                'confidence': f"{confidence:.2%}",
                'bbox': [x1, y1, x2, y2]
            })
        
        # Create summary message
        object_counts = {}
        for obj in detected_objects:
            class_name = obj['class']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        summary = []
        for class_name, count in object_counts.items():
            summary.append(f"{count} {class_name}{'s' if count > 1 else ''}")
        
        summary_text = f"Detected: {', '.join(summary)}"
        
        return image, summary_text, detected_objects
        
    except Exception as e:
        print(f"Processing error: {e}")
        return image, f"Error processing detection: {str(e)}", []

def predict_image(image):
    """
    Run YOLOv5 inference on the image
    """
    try:
        if model is None:
            return None, "Model not loaded", []
        
        # Run inference
        results = model(image, size=IMG_SIZE)
        
        # Process results and draw bounding boxes
        annotated_image, summary, detections = process_detection(image.copy(), results)
        
        return annotated_image, summary, detections
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, f"Error: {str(e)}", []

def pil_to_base64(image):
    """Convert PIL image to base64 string for display in web"""
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
        # Check if image was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file:
            # Read and process the image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Make prediction
            annotated_image, summary, detections = predict_image(image)
            
            if annotated_image is None:
                return jsonify({'error': summary})
            
            # Convert annotated image to base64 for display
            result_image_b64 = pil_to_base64(annotated_image)
            
            return jsonify({
                'summary': summary,
                'detections': detections,
                'annotated_image': result_image_b64,
                'total_objects': len(detections)
            })
            
    except Exception as e:
        print(f"Route error: {e}")
        return jsonify({'error': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
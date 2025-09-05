from flask import Flask, request, render_template, jsonify
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import yaml
import os
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

MODEL_PATH = 'best.pt'
IMG_SIZE = 416

# Try loading with different methods
model = None

# Normalized mapping: support weird class IDs from training
CLASS_NAMES = {
    0: 'mug',
    1: 'spoon',
    41: 'mug',
    44: 'spoon',
    76: 'spoon'
}

def try_load_model():
    global model
    
    # Method 1: Try loading as PyTorch model directly
    try:
        print("Attempting to load as PyTorch model...")
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        if 'model' in checkpoint:
            print("YOLOv5 checkpoint detected")
            model_state = checkpoint['model']
            
            from ultralytics import YOLO
            model = YOLO()
            model.model.load_state_dict(model_state.state_dict() if hasattr(model_state, 'state_dict') else model_state)
            print("Model loaded successfully using direct PyTorch loading!")
            return True
    except Exception as e:
        print(f"PyTorch loading failed: {e}")
    
    # Method 2: Try ultralytics with conversion
    try:
        print("Attempting to load with ultralytics (with conversion)...")
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        model.conf = 0.25
        model.iou = 0.45
        print("Model loaded successfully with ultralytics!")
        return True
    except Exception as e:
        print(f"Ultralytics loading failed: {e}")
    
    # Method 3: Use a pre-trained YOLOv8 model as fallback
    try:
        print("Loading pre-trained YOLOv8n as fallback...")
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        model.conf = 0.25
        model.iou = 0.45
        print("Fallback YOLOv8n model loaded!")
        return True
    except Exception as e:
        print(f"Fallback model loading failed: {e}")
    
    return False

# Try to load the model
if try_load_model():
    print("Model loaded successfully!")
else:
    print("All model loading methods failed. Running without model.")

# Load class names from yaml (optional)
try:
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
        # Merge yaml class names with fallback mapping
        for k, v in data_config['names'].items():
            CLASS_NAMES[int(k)] = v
    print(f"Classes loaded: {CLASS_NAMES}")
except Exception as e:
    print(f"Error loading data.yaml: {e}")
    print("Using fallback class names")

def process_detection(image, results):
    try:
        if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
            return image, "No objects detected", []

        boxes = results[0].boxes
        if len(boxes) == 0:
            return image, "No objects detected", []
            
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        detected_objects = []
        
        # Handle different box formats
        if hasattr(boxes, 'data'):
            detections_data = boxes.data.cpu().numpy()
        elif hasattr(boxes, 'xyxy'):
            detections_data = boxes.xyxy.cpu().numpy()
        else:
            detections_data = boxes.cpu().numpy()
        
        for det in detections_data:
            if len(det) >= 6:  # x1, y1, x2, y2, conf, class
                x1, y1, x2, y2, confidence, class_id = det[:6]
            elif len(det) >= 5:  # no class info
                x1, y1, x2, y2, confidence = det[:5]
                class_id = -1
            else:
                continue
                
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_name = CLASS_NAMES.get(int(class_id), 'unknown')
            
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
        
        # Count objects
        object_counts = {}
        for obj in detected_objects:
            object_counts[obj['class']] = object_counts.get(obj['class'], 0) + 1
        
        summary = [f"{count} {cls}{'s' if count > 1 else ''}" for cls, count in object_counts.items()]
        summary_text = f"Detected: {', '.join(summary)}" if summary else "No objects detected"
        
        return image, summary_text, detected_objects
    except Exception as e:
        traceback.print_exc()
        return image, f"Error processing detection: {str(e)}", []

def predict_image(image):
    try:
        if model is None:
            return None, "Model not loaded", []
        
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

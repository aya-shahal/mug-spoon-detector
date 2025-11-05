# ☕ Flask YOLOv5 Object Detection App

## 🧠 What This Model Does
This project is a **Flask-based web application** that performs real-time object detection using a **pre-trained YOLOv5 model**.  
At its core, it provides an API and a simple web interface. When you upload an image, the application uses your custom `best.pt` model to identify specific objects — in this case, **"mugs"** and **"spoons"** — and draws bounding boxes around them.  
It then returns the annotated image and a summary of the detected objects.

---

## ⚙️ Problem Statement
Manual image inspection or counting objects in photos is time-consuming and error-prone.  
This project uses AI-powered object detection to automate the process, providing instant and accurate visual results through a lightweight Flask web app.

---

## 🧩 Tech Stack
| Component | Purpose | Tool / Library |
|------------|----------|----------------|
| **YOLOv5** | Object detection model | Ultralytics YOLOv5 |
| **Flask** | Web server and API framework | Flask |
| **PyTorch** | Model inference | Torch |
| **OpenCV / Pillow** | Image processing | cv2 / PIL |

---

## 🚀 Local Setup (Without Docker)

### 1. Clone the YOLOv5 Repository
git clone https://github.com/ultralytics/yolov5.git  
cd yolov5  

### 2. Create and Activate a Virtual Environment
python -m venv venv  
source venv/bin/activate   # (Linux/macOS)  
venv\Scripts\activate      # (Windows)  

### 3. Install Dependencies
pip install -r requirements.txt  
pip install flask  

💡 If PyTorch doesn’t install automatically:  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  

---

### 4. Add Your Custom Model
Place your trained YOLOv5 model file (`best.pt`) inside the cloned `yolov5` directory.  
Example:  
yolov5/best.pt  

---

### 5. Run the Application
From inside the `yolov5` folder:  
python app.py  

Then open your browser and go to:  
http://127.0.0.1:5000  

Upload an image containing mugs or spoons to see the detected objects.

---

## 🔍 Output
- Annotated image saved in `/static/`  
- Console or webpage summary showing number of detected objects (e.g., “3 mugs, 1 spoon”)



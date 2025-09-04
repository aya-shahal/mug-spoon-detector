What This Model Does
This project is a Flask-based web application that performs real-time object detection using a pre-trained YOLOv5 model. At its core, it provides an API and a simple web interface. When you upload an image, the application uses your custom best.pt model to identify specific objects—in this case, "mugs" and "spoons"—and draws bounding boxes around them. It then returns the annotated image and a summary of the detected objects.

Local Setup (Without Docker)
You can run this application locally by setting up a Python environment.

Clone the YOLOv5 repository: Your model depends on the YOLOv5 library. You must clone the repository to run the model correctly.


git clone https://github.com/ultralytics/yolov5.git
Install dependencies: Navigate into the cloned repository and install all required packages. This includes torch, Flask, and others listed in the requirements.txt file.

cd yolov5
pip install -r requirements.txt

Run the application: From the yolov5 directory, start the Flask server.

python app.py

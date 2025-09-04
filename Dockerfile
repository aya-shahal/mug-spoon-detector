# Start with a base image that has Python and PyTorch
FROM pytorch/pytorch:latest

# Set a working directory
WORKDIR /app

# Copy your application code and all necessary files
COPY app.py .
COPY data.yaml .
COPY requirements.txt .
COPY templates/ templates/
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Install dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
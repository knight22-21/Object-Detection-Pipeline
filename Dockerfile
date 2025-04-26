# Base image with PyTorch + CUDA (change CUDA version as per your GPU)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements (you can customize this list)
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Set the default command
CMD ["python", "main.py"]

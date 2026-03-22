FROM python:3.11-slim

# System deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Match sandbox package versions
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    ultralytics==8.1.0 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.2.0 \
    numpy==1.26.4 \
    scikit-learn==1.4.0 \
    scipy==1.12.0 \
    safetensors==0.4.2

# No network after this point (simulated by not installing anything else)
WORKDIR /app

# Copy submission zip
COPY submission.zip /app/submission.zip
RUN python -c "import zipfile; zipfile.ZipFile('submission.zip').extractall('.')" && rm submission.zip

# Copy test images
COPY test_images/ /data/images/

# Run
CMD ["python", "run.py", "--input", "/data/images", "--output", "/output/predictions.json"]

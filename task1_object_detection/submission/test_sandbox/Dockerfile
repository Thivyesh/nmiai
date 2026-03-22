FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 ultralytics==8.1.0 \
    opencv-python-headless==4.9.0.80 Pillow==10.2.0 numpy==1.26.4 \
    scikit-learn==1.4.0 scipy==1.12.0 safetensors==0.4.2 \
    onnxruntime==1.20.0
WORKDIR /app
COPY submission.zip /app/submission.zip
RUN python -c "import zipfile; zipfile.ZipFile('submission.zip').extractall('.')" && rm submission.zip
COPY test_images/ /data/images/
RUN mkdir -p /output
CMD ["python", "run.py", "--input", "/data/images", "--output", "/output/predictions.json"]

FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch wheel first to cut down image size and avoid CUDA dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy runtime requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model artifacts
COPY web_ui/ ./web_ui/
COPY models/ ./models/
COPY thresholds.json* ./

# Configure Hugging Face Spaces required port (7860) and host
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=7860
EXPOSE 7860

CMD ["python", "web_ui/app.py"]

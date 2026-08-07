FROM python:3.11-slim

# Install system audio dependencies (ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose Hugging Face Space port
EXPOSE 7860

# Start Uvicorn REST API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

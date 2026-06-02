FROM python:3.11-slim

# Install system dependencies for MediaPipe and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into a folder named 'mmbi' so imports resolve perfectly
COPY . /app/mmbi

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860
EXPOSE 7860

# Start the server
CMD ["python", "-m", "mmbi.server"]

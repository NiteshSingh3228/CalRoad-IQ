FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (OpenCV requires libgl)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# For free tier limits, we exclude torch/geoclip (app falls back to AI Vision).
RUN grep -v "^torch\|^torchvision\|^geoclip" requirements.txt > slim_requirements.txt
RUN pip install --no-cache-dir -r slim_requirements.txt

# Copy source code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Project.py", "--server.port=8501", "--server.address=0.0.0.0"]

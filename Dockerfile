FROM python:3.11-slim

# Avoid Python output buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system packages needed by pdfminer.six
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Command is NOT run here (compose handles starting main.py)
CMD ["python", "app/main.py"]

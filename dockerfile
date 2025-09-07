# Minimal, reproducible Python image
FROM python:3.11-slim

# Don't buffer logs; safer for Railway logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system deps only if you later need them (kept tiny here)
RUN pip install --no-cache-dir --upgrade pip

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Start
CMD ["python", "-u", "main.py"]

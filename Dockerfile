# Multi-stage production container setup for ScaleRL API deployment
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a wheels directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal stage
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY scalerl_engine.py .
COPY scalerl_dashboard.html .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1


# Expose production ASGI gateway port
EXPOSE 8000

# Run FastAPI engine via uvicorn
CMD ["uvicorn", "scalerl_engine:app", "--host", "0.0.0.0", "--port", "8000"]
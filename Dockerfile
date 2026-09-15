# Multi-stage production container for the ScaleRL inference gateway.
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Train the policy at build time so the image ships with a ready artifact.
COPY scalerl ./scalerl
COPY train.py .
ENV PATH=/root/.local/bin:$PATH
RUN python train.py --steps 400000 --seed 42 --no-plot

# ---- Runtime stage ----
FROM python:3.11-slim AS runner

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/artifacts ./artifacts
COPY scalerl ./scalerl
COPY server.py .
COPY scalerl_dashboard.html .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health').status==200 else 1)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

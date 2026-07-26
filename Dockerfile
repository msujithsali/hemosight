# HemoSight multi-stage build — M Sujith Sali, ISE Dept, VTU Karnataka.
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=1729
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 libpango-1.0-0 libpangocairo-1.0-0 \
        libgdk-pixbuf-2.0-0 libffi-dev git \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip build && pip install --no-cache-dir ".[dev]"

FROM base AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
EXPOSE 8000
# Screening aid only — not a diagnostic device.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ src/
COPY data/ data/
COPY demo.py .
COPY .env.example .env.example

# Create outputs directory
RUN mkdir -p outputs

# Default command: run with a sample question
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["Explain RAG and why chunk overlap helps."]

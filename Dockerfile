# Traot - Docker Image
# ===========================
#
# Build: docker build -t ai-trade-bot .
# Run:   docker run -it ai-trade-bot
#
# For dashboard:
#   docker run -p 8501:8501 ai-trade-bot streamlit run dashboard.py

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add pytest for testing
RUN pip install --no-cache-dir pytest

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose dashboard port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command: run analysis
CMD ["python", "run_analysis.py"]

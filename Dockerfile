FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and project files
COPY src/ src/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -r -u 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default port for SSE mode
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pymilvus; print('OK')" || exit 1

# Default command
CMD ["mcp-server-milvus", "--sse", "--port", "8000"]
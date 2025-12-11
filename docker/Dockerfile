FROM python:3.11-slim

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy the Milvus MCP server source
COPY . .

# Install dependencies using uv
RUN uv pip install --system -e .

# Expose the SSE server port
EXPOSE 6280

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the MCP server in SSE mode
# Default to localhost:19530 but allow override via MILVUS_URI env var
CMD ["uv", "run", "src/mcp_server_milvus/server.py", "--sse", "--milvus-uri", "${MILVUS_URI:-http://localhost:19530}", "--port", "6280"]



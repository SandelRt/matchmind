FROM python:3.12-slim

# Install Node.js for Phoenix MCP + WC26 MCP (npx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-warm Phoenix MCP and WC26 MCP npx packages
RUN npx -y @arizeai/phoenix-mcp@latest --help 2>/dev/null || true
RUN npx -y wc26-mcp@latest --help 2>/dev/null || true

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Cloud Run listens on PORT env var (default 8080)
EXPOSE 8080

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]

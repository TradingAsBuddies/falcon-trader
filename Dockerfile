FROM python:3.11-slim

WORKDIR /app

# git is needed to pip-install falcon-core from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for better download handling
RUN pip install --upgrade pip

# Install falcon-core first (dependency)
RUN pip install --no-cache-dir --timeout 120 \
    "falcon-core @ git+https://github.com/TradingAsBuddies/falcon-core.git"

# Copy and install trader
COPY . .
RUN pip install --no-cache-dir .

# Dashboard runs on port 5000
ENV FLASK_HOST=0.0.0.0
EXPOSE 5000

# Default entrypoint is the orchestrator; override with CMD for dashboard
ENTRYPOINT ["falcon-trader"]

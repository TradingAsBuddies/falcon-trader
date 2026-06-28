# Fedora-native base (mandate: Fedora base OS for every part of the solution)
FROM registry.fedoraproject.org/fedora:42

WORKDIR /app

# git for pip git+ installs; gcc/python3-devel for any sdist builds
RUN dnf -y install python3 python3-pip python3-devel gcc git \
    && dnf clean all && rm -rf /var/cache/dnf

# Upgrade pip for better download handling
RUN pip install --upgrade pip

# Install falcon-core and psycopg2 for PostgreSQL support
RUN pip install --no-cache-dir --timeout 120 \
    "falcon-core[advisor,postgresql] @ git+https://github.com/TradingAsBuddies/falcon-core.git" \
    psycopg2-binary

# Copy and install trader
COPY . .
RUN pip install --no-cache-dir ".[youtube]"

# Signal/Risk tab (sangre-signal) reads AI keys at runtime, not baked here:
#   ANTHROPIC_API_KEY  (preferred) -> Claude narrative
#   PERPLEXITY_API_KEY (fallback)  -> Perplexity narrative
# Provide via runtime env (-e / compose env). If both are absent, the
# /api/risk-analysis structured flags still render; only the narrative
# degrades to the sangre-signal library text fallback.

# Dashboard runs on port 5000
ENV FLASK_HOST=0.0.0.0
EXPOSE 5000

# Default entrypoint is the orchestrator; override with CMD for dashboard
ENTRYPOINT ["falcon-trader"]

# Minimal CPU image for reproducible runs / sharing with the community.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[remote]"

# Default: run the synthetic benchmark and print metrics.
ENTRYPOINT ["scroll-anchor"]
CMD ["benchmark", "--output", "/app/results/bench"]

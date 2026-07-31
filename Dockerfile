# Minimal CPU-only image for the scroll-anchor CLI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Only what the package build and the CLI need at runtime.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# Normal (non-editable) install with the extras used by the public workflows.
# No dev extra: pytest is deliberately not installed.
RUN pip install --no-cache-dir ".[remote,benchmark,render]"

ENTRYPOINT ["scroll-anchor"]
CMD ["--help"]

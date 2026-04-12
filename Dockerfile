FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir openenv-core gymnasium pydantic fastapi uvicorn openai

# Copy the entire project structure
COPY email_triage_env /app/email_triage_env
COPY inference.py /app/inference.py
COPY README.md /app/README.md

# Set PYTHONPATH to root so we can import the module
ENV PYTHONPATH=/app

# Expose port 8000 for the environment server
EXPOSE 8000

# Default command: run the environment server
CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

# Create a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files and install
COPY --chown=user email_triage_env/pyproject.toml /app/email_triage_env/pyproject.toml
RUN pip install --no-cache-dir openenv-core gymnasium pydantic fastapi uvicorn openai

# Copy the entire project structure
COPY --chown=user . /app

# Set PYTHONPATH to root so we can import the module
ENV PYTHONPATH=/app

# Expose port 7860 for the environment server
EXPOSE 7860

# Default command: run the environment server on port 7860
CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]

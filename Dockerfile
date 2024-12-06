FROM python:3.11-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev

# Copy application code
COPY src/ ./src/

# Cloud Run will set PORT environment variable
ENV PORT=8090

# Run the application
CMD exec poetry run uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
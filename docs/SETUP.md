# Setup Guide

## Prerequisites
- Python 3.8+
- Google Cloud account
- OpenAI API key

## Installation

1. Clone the repository:
git clone <repository-url>
cd <project-directory>


2. Install dependencies with Poetry:
bash
Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -
Install dependencies
poetry install
Activate virtual environment
poetry shell


3. Set up environment variables:

export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

export OPENAI_API_KEY=your_key_here

4. Configure Google Cloud Storage:
- Create a bucket
- Update bucket name in config
- Set up authentication

## Development Setup
bash
Run with hot reload
uvicorn app.main:app --reload --port 8090

## Production Deployment
1. Build Docker image
2. Deploy to Cloud Run
3. Set environment variables in Cloud Run
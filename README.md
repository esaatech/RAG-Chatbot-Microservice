# Document Query API

A FastAPI-based service that enables document processing and intelligent querying using RAG (Retrieval Augmented Generation) technology. The system allows users to upload documents, process them into vector embeddings, and perform natural language queries against the document content.

## Features

- 📄 Support for multiple document formats (PDF, TXT, DOCX, CSV)
- 🔍 RAG-based document querying
- ⚡ Real-time document processing
- 🎯 Configurable response parameters
- 🔄 Document and configuration updates
- 🗑️ Document deletion capability
- 📊 Cache statistics monitoring

## Tech Stack

- FastAPI
- OpenAI
- Google Cloud Storage
- Docker
- Poetry for dependency management

## Quick Start

### Prerequisites

- Python 3.8+
- Google Cloud account
- OpenAI API key
- Poetry (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies:
```bash
poetry install
poetry shell
```

3. Set up environment variables:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
export OPENAI_API_KEY=your_key_here
```

### Running Locally

```bash
uvicorn app.main:app --reload --port 8090
```

## API Endpoints

### Document Management

- `POST /documents/upload` - Upload and process new documents
- `POST /documents/{key}/query` - Query processed documents
- `PUT /documents/{key}/update` - Update existing documents
- `PUT /documents/{key}/update-config` - Update document configurations
- `DELETE /documents/{key}` - Delete documents
- `GET /documents/cache-stats` - Get cache statistics

For detailed API documentation, see [API.md](docs/API.md)

## Deployment

The application is configured for deployment to Google Cloud Run. See [SETUP.md](docs/SETUP.md) for detailed deployment instructions.

## Configuration

Documents can be processed with custom configurations including:
- Company name
- Agent role
- Response style
- Tone

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request




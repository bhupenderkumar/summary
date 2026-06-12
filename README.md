# File Agent

A FastAPI-based RAG (Retrieval-Augmented Generation) application that lets users upload files, generate embeddings, and chat with their data using LLMs via OpenRouter.

## Features

- **File Upload** – Upload CSV or text files for analysis
- **Chunked Embeddings** – Splits large files into chunks and generates vector embeddings using OpenAI's `text-embedding-3-small` model via OpenRouter
- **Semantic Search** – Finds relevant chunks using cosine similarity
- **Chat with Data** – Ask questions about uploaded data; responses are grounded in the most relevant file content
- **User Filtering** – Filter CSV data by `user_id` for targeted analysis
- **Web UI** – Built-in HTML interface for uploading files and chatting
- **AWS Lambda Ready** – Deployable as a serverless function using Mangum

## Setup

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

```bash
pip install -r requirement.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
key=your_openrouter_api_key
```

### Running Locally

```bash
uvicorn main:app --reload
```

The app will be available at `http://localhost:8000`.

## API Endpoints

| Method | Path      | Description                              |
|--------|-----------|------------------------------------------|
| GET    | `/`       | Serves the web UI                        |
| POST   | `/upload` | Upload a file for chunking and embedding |
| GET    | `/users`  | List extracted user IDs from CSV data    |
| POST   | `/chat`   | Send a message and get an AI response    |

### POST `/chat` Request Body

```json
{
  "message": "Summarize the network traffic",
  "user_id": ""
}
```

## Project Structure

```
├── main.py            # FastAPI app with upload, chat, and embedding logic
├── llm_provider.py    # LLM client wrapper (OpenRouter)
├── prompt.txt         # System prompt template
├── requirement.txt    # Python dependencies
├── templates/
│   └── index.html     # Web UI
└── test_data/
    └── banking_network_traffic.csv
```

## Deployment

The app includes Mangum integration for deployment to AWS Lambda behind API Gateway.

# PDF RAG Assistant

A simple Retrieval-Augmented Generation app that allows users to upload a PDF and ask questions grounded in the document.

## Features

- Upload PDF files
- Split PDF text into overlapping chunks
- Generate embeddings
- Store and search document chunks using Chroma
- Ask questions about the PDF
- Display retrieved source chunks with page numbers

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- OpenAI API
- PyPDF

## How It Works

1. The user uploads a PDF.
2. The app reads the PDF and splits it into text chunks.
3. Each chunk is converted into an embedding vector.
4. The embeddings are stored in a Chroma vector database.
5. When the user asks a question, the app retrieves the most relevant chunks.
6. The retrieved context is passed to an LLM to generate a grounded answer.

## Setup

```bash
pip install -r requirements.txt
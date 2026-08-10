# EvidenceRAG Chatbot

Conversational Retrieval-Augmented Generation (RAG) assistant that answers questions using only the content of indexed documents. This project is a functional local MVP for exploring document ingestion, semantic retrieval, grounded generation and LLM observability.

## Features

- Ingests TXT files and individual PDF files.
- Splits documents into configurable chunks.
- Generates local embeddings with `all-MiniLM-L6-v2`.
- Stores vectors in persistent ChromaDB.
- Retrieves the most relevant document fragments.
- Generates grounded answers through an OpenAI-compatible LLM API.
- Shows source fragments in a Chainlit web interface.
- Captures user feedback and traces with LangSmith.
- Includes an evaluation workflow for retrieval precision, recall and answer relevance.

## Architecture

```text
TXT/PDF -> ingestion -> chunks -> local embeddings -> ChromaDB
                                                       |
Question -> similarity search -> context prompt -> LLM -> answer + sources
                                                       |
                                             LangSmith traces and feedback
```

## Technology

Python, LangChain, Chainlit, ChromaDB, Sentence Transformers, Hugging Face, LangSmith and OpenAI-compatible APIs.

## Local setup

Requirements: Python 3.13 recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
```

Set the variables in `.env`. Never commit `.env`, API keys or private documents.

Create the vector index:

```bash
python ingest.py demo.txt
```

Run a terminal query:

```bash
python query.py "¿Cuántas horas duerme un gato?"
```

Run the web interface:

```bash
chainlit run app.py -w
```

Run the evaluation workflow:

```bash
python eval.py
```

## Project structure

```text
app.py          Chainlit application, sources and feedback
ingest.py       Loading, chunking, embeddings and indexing
query.py        Command-line retrieval and generation
eval.py         Dataset creation and evaluation workflow
demo.txt        Sample document
requirements.txt
```

## Current status

This is a functional local MVP. It is not a production product and does not yet index files uploaded dynamically through the Chainlit interface. The current evaluation workflow is intended for iterative technical validation and experimentation.

## Security

The `.env` file, ChromaDB index, caches and API keys are excluded from Git. If a credential has ever been exposed, revoke and regenerate it before publishing the repository.

## License

MIT. See `LICENSE`.

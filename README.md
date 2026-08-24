# Document Ingestion RAG (LangChain)

End-to-end retrieval-augmented generation (RAG) over an HR policy PDF.

The pipeline loads a PDF from a URL with Docling, splits it into chunks, embeds those chunks with a Hugging Face sentence-transformer, stores them in Qdrant (in-memory), retrieves the most similar chunks for a question, and asks Groq to answer using only that context.

## Pipeline

1. **Load** — `DoclingLoader` fetches and parses the PDF.
2. **Split** — `RecursiveCharacterTextSplitter` creates overlapping chunks.
3. **Embed** — `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) encodes each chunk.
4. **Store** — `QdrantVectorStore` keeps vectors in an in-memory collection.
5. **Retrieve** — similarity search returns the top-k chunks.
6. **Generate** — Groq answers the question using the retrieved context only.

Configuration and results are typed with Pydantic (`IngestConfig`, `RAGQuery`, `RAGResult`, `GroqSettings`).

## Requirements

- Python 3.10 or later
- A [Groq API key](https://console.groq.com/keys)

The first run downloads the embedding model from Hugging Face (network required).

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`langchain-docling` only installs a slim Docling core. The `docling` package adds PDF backends (`pypdfium2`, `docling-parse`, and related models) required by `DoclingLoader`.

Copy the example environment file and add your Groq key:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Do not commit `.env`. `.env.example` is safe to share.

## Run

```bash
python app.py
```

This ingests the sample AtliqAI HR policies PDF and asks:

> What is the company's policy on remote work?

It prints the model answer and the source chunks used as context.

## Project layout

| File | Purpose |
|------|---------|
| `app.py` | Ingest, retrieve, and generate pipeline |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for Groq credentials |
| `.env` | Local secrets (gitignored) |

## Configuration

Defaults live in `IngestConfig` and `RAGQuery` in `app.py`:

| Setting | Default |
|---------|---------|
| Source PDF | AtliqAI HR policies sample on GitHub |
| Chunk size / overlap | 1000 / 200 characters |
| Embedding model | `all-MiniLM-L6-v2` |
| Qdrant | `:memory:` collection `atliqai_hr_policies` |
| Retrieval `top_k` | 3 |
| Chat model | `GROQ_MODEL` or `openai/gpt-oss-20b` |

Change the source URL, query, or chunking by editing `main()` (or the Pydantic models it constructs).

## Troubleshooting

**`ModuleNotFoundError: No module named 'pypdfium2'`**

Docling's PDF parser is not in the slim extra pulled by `langchain-docling`. Install the full package (already listed in `requirements.txt`):

```bash
pip install -r requirements.txt
# or
pip install "docling==2.121.0"
```

Then run `python app.py` again.

## License

Use and share as you like for learning and demos.

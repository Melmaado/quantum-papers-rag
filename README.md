# Quantum Paper RAG

A retrieval-augmented generation (RAG) assistant that answers questions about a small corpus of quantum physics papers from arXiv, citing the paper and section each answer comes from.

> **Status:** work in progress.

## Goal

The aim is not a polished chatbot, but a pipeline where every design choice is measured. The project compares three retrieval strategies on a hand-written test set:

- **Vector search** with sentence embeddings
- **Keyword search** with BM25
- **Hybrid search**, combining both with Reciprocal Rank Fusion

## Pipeline

1. **Ingestion:** download the LaTeX source of ~15 arXiv papers and convert it to text, keeping section boundaries.
2. **Chunking:** split each section into ~200-token chunks with overlap, tagged with paper ID and section name.
3. **Indexing:** embed chunks with `sentence-transformers` and build a BM25 index.
4. **Retrieval:** vector, BM25 or hybrid, returning the top 5 chunks.
5. **Generation:** an LLM answers using only the retrieved chunks, cites `[arXiv ID, section]`, and says so when the answer is not in the corpus.

## Evaluation

A set of 15 test questions with known source sections:

- exact-jargon questions (where BM25 should do well)
- reworded questions (where embeddings should do well)
- cross-paper questions
- out-of-corpus questions (which should be refused)

| Mode | Hit@5 | Correct answers | Correct citations |
| --- | --- | --- | --- |
| Vector | – | – | – |
| BM25 | – | – | – |
| Hybrid | – | – | – |

*Results coming soon.*

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

An OpenAI API key is needed for the generation step, set as the `OPENAI_API_KEY` environment variable.

## Data

The papers themselves are **not included** in this repository, as most arXiv papers are not licensed for redistribution. The repository only contains the list of arXiv IDs and the script that downloads them.

Thank you to arXiv for use of its open access interoperability.
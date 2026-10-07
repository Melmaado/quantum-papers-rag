# Quantum Paper RAG

A retrieval-augmented generation (RAG) assistant that answers questions about a small corpus of quantum physics papers from arXiv, citing the paper and section each answer comes from.

> **Status:** work in progress. The full pipeline runs end to end; the evaluation is being built.

## Goal

The aim is not a polished chatbot, but a pipeline where every design choice is measured. The project compares three retrieval strategies on a test set:

- **Vector search** with sentence embeddings
- **Keyword search** with BM25
- **Hybrid search**, combining both with Reciprocal Rank Fusion (RRF)

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| 1. Download | `download.py` | Downloads the LaTeX source of the 15 arXiv papers listed in `papers.txt`. |
| 2. Extraction | `extract.py` | Finds the main `.tex` file, removes comments and inlines `\input` / `\include` files. |
| 3. Chunking | `chunk.py` | Splits each paper by section and subsection, converts LaTeX to plain text and cuts it into 150-word chunks with a 20-word overlap, each tagged with its arXiv ID and section path. |
| 4. Retrieval | `retrieve.py` | Embeds the chunks with `all-MiniLM-L6-v2` (cosine similarity with NumPy), builds a BM25 index and returns the top 5 chunks with vector, BM25 or hybrid search. |
| 5. Generation | `ask.py` | An LLM answers using only the retrieved chunks, cites `[arXiv ID, section]`, and says so when the answer is not in the corpus. |

## Evaluation

The test set combines two sources:

- **3 hand-written questions** (`eval/questions.json`): one exact-jargon question, one reworded question and one out-of-corpus question, which should be refused.
- **27 synthetic questions** (`eval/synthetic_questions.json`, made by `generate_questions.py`): an LLM writes one question from a randomly sampled chunk, alternating between questions that reuse the passage's technical terms (14) and reworded questions that avoid them (13).

Each question lists its expected source sections. A retrieval counts as a hit when one of the top 5 chunks comes from an expected paper and section (`evaluate.py`).

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

An OpenAI API key is needed for generation and for the synthetic questions. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

Then run the scripts in order:

```bash
python download.py
python extract.py
python chunk.py
python retrieve.py   # also computes and caches the embeddings
python ask.py
```

## Data

The papers themselves are **not included** in this repository, as most arXiv papers are not licensed for redistribution. The repository only contains the list of arXiv IDs and the script that downloads them.

Thank you to arXiv for use of its open access interoperability.
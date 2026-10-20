# Quantum Paper RAG

A retrieval-augmented generation (RAG) assistant that answers questions about a small corpus of quantum physics papers from arXiv, citing the paper and section each answer comes from.

> **Status:** work in progress. The pipeline runs end to end and retrieval is evaluated; answer quality and refusals are not evaluated yet.

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
- **27 synthetic questions** (`eval/synthetic_questions.json`, made by `generate_questions.py`): an LLM writes one question from a randomly sampled chunk, alternating between questions that reuse the passage's technical terms (`lingo`, 14) and reworded questions that avoid them (`reworded`, 13). Questions were reviewed by hand.

Each question lists its expected source sections. `evaluate.py` reports two metrics:

- **Section-level Hit@5**: one of the top 5 chunks comes from an expected paper **and** section.
- **Paper-level Hit@5**: one of the top 5 chunks comes from an expected paper, whatever its section.

The 3 hand-written questions are not part of these results yet.

### Results (Hit@5, synthetic questions)

| Mode | Section, all (27) | Section, lingo (14) | Section, reworded (13) | Paper, all (27) |
| --- | --- | --- | --- | --- |
| Vector | 66.7% (18) | 92.9% (13) | 38.5% (5) | 85.2% (23) |
| BM25 | **74.1%** (20) | **100%** (14) | **46.2%** (6) | **92.6%** (25) |
| Hybrid | 70.4% (19) | 92.9% (13) | **46.2%** (6) | **92.6%** (25) |

**Takeaways**

- The gap between modes is small: one question each, out of 27. With this sample size, no mode is clearly better.
- The real gap is between question types: about 95% on `lingo` questions against about 40% on `reworded` ones, whatever the mode. The weak point is questions phrased with different words than the papers.
- Most misses are near misses: at paper level, hybrid search finds the right paper for 25 questions out of 27. It lands on the wrong paper only twice.

### Error analysis

Every failed question was reviewed by hand (`evaluate.py` prints the expected and retrieved sections). Failures fall into four groups:

1. **Right paper, neighbouring section** (~4 questions): the hit criterion requires the exact section, and retrieval lands next to it.
2. **Likely false negatives** (~2): the answer also appears in another section that was retrieved, but each synthetic question has a single expected section.
3. **Real misses** (~3): reworded questions removed every anchoring term (e.g. "enlarged arrangement" for *inflation*), and neither embeddings nor BM25 bridge the gap.
4. **Duplicate sections**: several chunks from the same section can fill the top 5, leaving less room for other sections.

### Next steps

- Evaluate the 3 hand-written questions, including the out-of-corpus refusal.
- Keep only the best chunk per section in the results.
- Rewrite the question with an LLM before retrieval, to map plain-language questions back to the papers' terms.
- Evaluate the answers themselves (correctness and citations) and the refusals on out-of-corpus questions.

## Known limitations

- Section headings containing braces are truncated by the regex (e.g. `Proof of Theorem~\ref*{thm:main`).
- Author names cited with `\citeauthor` are lost during LaTeX conversion, which can leave fragments like "(EPR, )".
- The "Main text" chunk of each paper still contains author affiliations.
- `\today` is converted to the date the pipeline was run.
- Inline LaTeX comments at the end of a line are kept.
- Synthetic questions have a single expected section, which can create false negatives.
- 27 questions is a small sample: one question weighs 3.7 points.

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
python evaluate.py   # Hit@5 per mode, per question type and per paper, plus failed questions
```

## Data

The papers themselves are **not included** in this repository, as most arXiv papers are not licensed for redistribution. The repository only contains the list of arXiv IDs and the script that downloads them.

Thank you to arXiv for use of its open access interoperability.
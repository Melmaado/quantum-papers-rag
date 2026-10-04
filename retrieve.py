from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
import re

CHUNKS_PATH = Path("data") / "chunks.json"
EMBEDDINGS_PATH = Path("data") / "embeddings.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")
with open(CHUNKS_PATH, "r", encoding="utf-8") as j:
    chunks = json.load(j)

def search_vector(question, embeddings, chunks, k=5):
    """Return the k chunks most similar to the question, as (score, chunk) pairs.
    The question is encoded with the same model as the chunks. Since all
    embeddings are normalized, the dot product equals the cosine similarity.
    Results are sorted from the highest score to the lowest.
    """
    best_of = []
    question_vector = model.encode(question, normalize_embeddings = True)
    scores = embeddings @ question_vector
    top = np.argsort(scores)[::-1][:k]
    for pos in top:
        best_of.append((scores[pos],chunks[pos]))
    return best_of


def tokenize(text):
    """Lowercase the text and split it into words, dropping punctuation."""
    text = text.lower()
    text = re.findall(r"\w+",text)
    return text

def search_bm25(question, bm25, chunks, k=5):
    """Return the k chunks with the highest BM25 score for the question, as
    (score, chunk). The question is tokenized like the chunks. Unlike cosine
    similarities, BM25 are not bounded between 0 and 1, they are sums over
    the question's words."""
    best_of = []
    tokenized_question = tokenize(question)
    scores = bm25.get_scores(tokenized_question)
    top = np.argsort(scores)[::-1][:k]
    for pos in top:
        best_of.append((scores[pos],chunks[pos]))
    return best_of

if not EMBEDDINGS_PATH.exists():
    texts =[]
    for chunk in chunks:
        texts.append(chunk["text"])

    embeddings = model.encode(texts,show_progress_bar = True, normalize_embeddings = True)
    np.save(EMBEDDINGS_PATH, embeddings)
else:
    embeddings = np.load(EMBEDDINGS_PATH)

tokenized_chunks = []
for chunk in chunks:
    tokenized_chunks.append(tokenize(chunk["text"]))

bm25 = BM25Okapi(tokenized_chunks)

print(embeddings.shape)

results = search_vector("What is the CHSH inequality?", embeddings, chunks)
for score, chunk in results:
   print(f"{score:.3f}: {chunk['paper_id']} - {chunk['section']} - {chunk['text']} \n")

print("=================BM25=================")

results = search_bm25("What is the CHSH inequality?", bm25, chunks)
for score, chunk in results:
   print(f"{score:.3f}: {chunk['paper_id']} - {chunk['section']} - {chunk['text']} \n")


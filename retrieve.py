from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from text_utils import tokenize

CHUNKS_PATH = Path("data") / "chunks.json"
EMBEDDINGS_PATH = Path("data") / "embeddings.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")
with open(CHUNKS_PATH, "r", encoding="utf-8") as j:
    chunks = json.load(j)


def rank_vector(question, embeddings, depth):
    """Return the positions of the `depth` chunks most similar to the question, best first.
    Since all embeddings are normalized, the dot product equals the cosine similarity.
    """
    question_vector = model.encode(question, normalize_embeddings = True)
    scores = embeddings @ question_vector
    return np.argsort(scores)[::-1][:depth]

def dedupe_by_section(positions, chunks, k):
    """Return up to k chunks from the ranked positions, at most one per section.
    Positions are read in ranking order. A chunk is skipped when another chunk
    from the same (paper_id, section) was already kept, so the results cover
    more distinct sections. Returns fewer than k chunks if there are not enough
    distinct sections.
    """
    results=[]
    seen= set()
    for position in positions:
        chunk = chunks[position]
        pair = (chunk['paper_id'],chunk['section'])
        if pair in seen:
            continue
        else:
            results.append(chunk)
            seen.add(pair)
            if len(results)==k:
                return results
    return results

fake_chunks = [
{"paper_id": "A", "section": "Intro", "text": "..."},
{"paper_id": "A", "section": "Intro", "text": "..."},
{"paper_id": "B", "section": "Intro", "text": "..."},
{"paper_id": "A", "section": "Methods", "text": "..."},
]

assert dedupe_by_section([0, 1, 2, 3], fake_chunks, 3) == [fake_chunks[0], fake_chunks[2], fake_chunks[3]]

def search_vector(question, embeddings, chunks, k=5, dedupe=False, depth=20):
    """Return the k chunks most similar to the question, best first, using vector search.
    With dedupe=True, keeps at most one chunk per section, chosen among the top depth candidates.
    """
    if dedupe:
        positions = rank_vector(question, embeddings, depth)
        return dedupe_by_section(positions, chunks, k)
    best_of = []
    top = rank_vector(question, embeddings, k)
    for pos in top:
        best_of.append(chunks[pos])
    return best_of


def rank_bm25(question, bm25, depth):
    """Return the positions of the `depth` chunks with the highest BM25 score, best first.
    The question is tokenized like the chunks.
    """
    tokenized_question = tokenize(question)
    scores = bm25.get_scores(tokenized_question)
    return np.argsort(scores)[::-1][:depth]

def search_bm25(question, bm25, chunks, k=5, dedupe=False, depth=20):
    """Return the k chunks with the highest BM25 score for the question, best first.
    With dedupe=True, keeps at most one chunk per section, chosen among the top depth candidates.
    """
    if dedupe:
        positions = rank_bm25(question, bm25, depth)
        return dedupe_by_section(positions, chunks, k)
    best_of = []
    top = rank_bm25(question, bm25, k)
    for pos in top:
        best_of.append(chunks[pos])
    return best_of

def rrf_fuse(rankings, k=60):
    """Fuse several rankings with Reciprocal Rank Fusion (RRF).
    Each ranking is a list of items, best first. An item at rank r
    (starting at 1) gets 1 / (k + r) from that ranking, and its scores
    are summed over all rankings. Returns the items sorted by fused
    score, best first. Items found in several rankings rise to the top.
    """
    rrf_scores = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            rrf_scores[position] = rrf_scores.get(position,0) + 1/(k+rank)
    return sorted(rrf_scores, key=rrf_scores.get, reverse=True)

assert rrf_fuse([["A","B","C"],["C","D","A"]]) == ['A', 'C', 'B', 'D']

def search_hybrid(question, embeddings, bm25, chunks, k = 5, depth = 20, dedupe=False):
    """Return the k best chunks for the question, combining vector and BM25 search.
    Each method ranks the chunks, and the top `depth` positions of both
    rankings are fused with Reciprocal Rank Fusion. Chunks ranked well by
    both methods rise to the top, even if neither ranks them first.
    Returns a list of chunks, without scores.
    With dedupe=True, keeps at most one chunk per section, chosen among the top depth candidates.
    """
    best_of = []
    rankings = []
    top_vector = rank_vector(question, embeddings, depth)
    top_bm25 = rank_bm25(question, bm25, depth)

    rankings.append(top_vector)
    rankings.append(top_bm25)

    if dedupe:
        positions = rrf_fuse(rankings)
        return dedupe_by_section(positions, chunks, k)

    top_positions = rrf_fuse(rankings)[:k]

    for pos in top_positions:
        best_of.append(chunks[pos])

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

if __name__ == "__main__":
    print(embeddings.shape)

    print("=================VECTOR=================")

    results = search_vector("What is the CHSH inequality?", embeddings, chunks)
    for chunk in results:
        print(f"{chunk['paper_id']} - {chunk['section']} - {chunk['text'][:150]} \n")

    print("=================BM25=================")

    results = search_bm25("What is the CHSH inequality?", bm25, chunks)
    for chunk in results:
        print(f"{chunk['paper_id']} - {chunk['section']} - {chunk['text'][:150]} \n")

    print("=================HYBRID=================")

    results = search_hybrid("What is the CHSH inequality?", embeddings, bm25, chunks)
    for chunk in results:
        print(f"{chunk['paper_id']} - {chunk['section']} - {chunk['text'][:150]} \n")
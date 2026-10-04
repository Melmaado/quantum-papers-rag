from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import numpy as np

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


if not EMBEDDINGS_PATH.exists():
    texts =[]
    for chunk in chunks:
        texts.append(chunk["text"])

    embeddings = model.encode(texts,show_progress_bar = True, normalize_embeddings = True)
    np.save(EMBEDDINGS_PATH, embeddings)
else:
    embeddings = np.load(EMBEDDINGS_PATH)

print(embeddings.shape)

results = search_vector("What is the CHSH inequality?", embeddings, chunks)
for score, chunk in results:
    print(f"{score:.3f}: {chunk['paper_id']} - {chunk['section']} - {chunk['text']} \n")
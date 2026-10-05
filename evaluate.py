from retrieve import search_hybrid, embeddings, bm25, chunks
import json
from pathlib import Path

SYNTHETIC_QUESTIONS_PATH = Path("eval") / "synthetic_questions.json"

def is_hit(results, expected):
    """Return True if any retrieved chunk comes from one of the expected passage.
    there is a match between a chunk and a passage when both the paper_id and the section are equal
    """
    for chunk in results:
        for expectation in expected:
            if chunk['paper_id'] == expectation['paper_id'] and chunk['section'] == expectation['section']:
                return True
    return False

with open(SYNTHETIC_QUESTIONS_PATH, "r", encoding="utf-8") as j:
    questions = json.load(j)

question = questions[0]

results = search_hybrid(question['question'], embeddings, bm25, chunks)
print(is_hit(results, question['expected']))
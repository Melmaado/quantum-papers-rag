from retrieve import search_hybrid, search_vector, search_bm25, embeddings, bm25, chunks
import json
from pathlib import Path

SYNTHETIC_QUESTIONS_PATH = Path("eval") / "synthetic_questions.json"

def is_hit(results, expected):
    """Return True if any retrieved chunk comes from one of the expected passages.
    there is a match between a chunk and a passage when both the paper_id and the section are equal
    """
    for chunk in results:
        for expectation in expected:
            if chunk['paper_id'] == expectation['paper_id'] and chunk['section'] == expectation['section']:
                return True
    return False

with open(SYNTHETIC_QUESTIONS_PATH, "r", encoding="utf-8") as j:
    questions = json.load(j)

hits_vector = 0
hits_bm25 = 0
hits_hybrid = 0

for question in questions:
    if is_hit(search_vector(question['question'], embeddings, chunks), question['expected']): hits_vector+=1
    if is_hit(search_bm25(question['question'], bm25, chunks), question['expected']): hits_bm25 +=1
    if is_hit(search_hybrid(question['question'], embeddings, bm25, chunks), question['expected']): hits_hybrid +=1

hit_at_5_vector = 100*hits_vector/len(questions)
hit_at_5_bm25 = 100*hits_bm25/len(questions)
hit_at_5_hybrid = 100*hits_hybrid/len(questions)

print(f"""hit@5 vector = {hit_at_5_vector:.1f} ({hits_vector} questions out of {len(questions)})
hit@5 bm25 = {hit_at_5_bm25:.1f} ({hits_bm25} questions out of {len(questions)})
hit@5 hybrid = {hit_at_5_hybrid:.1f} ({hits_hybrid} questions out of {len(questions)})
""")

hits_lingo_vector = 0
hits_lingo_bm25 = 0
hits_lingo_hybrid = 0
hits_reworded_vector = 0
hits_reworded_bm25 = 0
hits_reworded_hybrid = 0
total_lingo = 0
total_reworded = 0

for question in questions:
    if question['type']=='lingo':
        total_lingo+=1
        if is_hit(search_vector(question['question'], embeddings, chunks), question['expected']): hits_lingo_vector+=1
        if is_hit(search_bm25(question['question'], bm25, chunks), question['expected']): hits_lingo_bm25 +=1
        if is_hit(search_hybrid(question['question'], embeddings, bm25, chunks), question['expected']): hits_lingo_hybrid +=1
        
    elif question['type']=='reworded':
        total_reworded+=1
        if is_hit(search_vector(question['question'], embeddings, chunks), question['expected']): hits_reworded_vector+=1
        if is_hit(search_bm25(question['question'], bm25, chunks), question['expected']): hits_reworded_bm25 +=1
        if is_hit(search_hybrid(question['question'], embeddings, bm25, chunks), question['expected']): hits_reworded_hybrid +=1

hit_lingo_vector = 100*hits_lingo_vector/total_lingo
hit_lingo_bm25 = 100*hits_lingo_bm25/total_lingo
hit_lingo_hybrid = 100*hits_lingo_hybrid/total_lingo
hit_reworded_vector = 100*hits_reworded_vector/total_reworded
hit_reworded_bm25 = 100*hits_reworded_bm25/total_reworded
hit_reworded_hybrid = 100*hits_reworded_hybrid/total_reworded


print(f"""lingo:
hit@5 vector = {hit_lingo_vector:.1f} ({hits_lingo_vector} questions out of {total_lingo})
hit@5 bm25 = {hit_lingo_bm25:.1f} ({hits_lingo_bm25} questions out of {total_lingo})
hit@5 hybrid = {hit_lingo_hybrid:.1f} ({hits_lingo_hybrid} questions out of {total_lingo})

reworded:
hit@5 vector = {hit_reworded_vector:.1f} ({hits_reworded_vector} questions out of {total_reworded})
hit@5 bm25 = {hit_reworded_bm25:.1f} ({hits_reworded_bm25} questions out of {total_reworded})
hit@5 hybrid = {hit_reworded_hybrid:.1f} ({hits_reworded_hybrid} questions out of {total_reworded})
""")
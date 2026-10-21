from retrieve import search_hybrid, embeddings, bm25, chunks
from ask import answer
from pathlib import Path
import json

SYNTHETIC_QUESTIONS_PATH = Path("eval")/"synthetic_questions.json"

with open(SYNTHETIC_QUESTIONS_PATH,"r",encoding="utf-8") as j:
    synthetic_questions = json.load(j)

questions = []

for question in synthetic_questions:
    if "three-party setup with no measurement" in question['question']:
        questions.append(question)
    if "What safeguards are recommended" in question['question']:
        questions.append(question)
    if "what family of numerical procedures" in question['question']:
        questions.append(question)

for question in questions:
    print(80*"=")
    
    print(f"- Question: {question['question']}")
    response = answer(question['question'])
    print(f"- Answer: {response}")
    
    results = search_hybrid(question['question'], embeddings, bm25, chunks)
    print("=== RETRIEVED ===")    
    for n, chunk in enumerate(results, 1):
        print(f"[{n}] - {chunk['paper_id']} - {chunk['section']}\n{chunk['text']}")
    
    print("=== EXPECTED ===")

    for expectation in question['expected']:
        for chunk in chunks:
            if chunk['paper_id'] == expectation['paper_id'] and chunk['section'] == expectation['section']:
                print(chunk['text'])
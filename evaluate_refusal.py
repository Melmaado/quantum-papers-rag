from ask import answer, REFUSAL
from pathlib import Path
import json

QUESTIONS_PATH = Path("eval") / "questions.json"
with open(QUESTIONS_PATH, "r", encoding="utf-8") as j:
    questions = json.load(j)

refusal_counter = 0
out_corpus = 0

for question in questions:
    if question['type']=="out_of_corpus":
        out_corpus+=1
        response = answer(question['question'])
        if REFUSAL in response:
            refusal_counter+=1
        else:
            print(question['question'])
            print(response)

print(f"""Refusal: {refusal_counter}/{out_corpus}""")
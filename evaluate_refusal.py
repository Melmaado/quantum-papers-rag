from ask import answer
from prompts import REFUSAL
from pathlib import Path
import json

QUESTIONS_PATH = Path("eval") / "questions.json"
SYN_QUESTIONS_PATH = Path("eval") / "synthetic_questions.json"

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


with open(SYN_QUESTIONS_PATH, "r", encoding="utf-8") as j:
    syn_questions = json.load(j)


syn_lingo = 0
syn_reworded = 0
syn_refusal_lingo = 0
syn_refusal_reworded = 0

for syn_question in syn_questions:
    if syn_question['type']=="lingo":
        syn_lingo+=1
    elif syn_question['type']=="reworded":
        syn_reworded+=1
    syn_response = answer(syn_question['question'])

    if REFUSAL in syn_response:
        if syn_question['type']=="lingo":
            syn_refusal_lingo+=1
        elif syn_question['type']=="reworded":
            syn_refusal_reworded+=1
        print(syn_question['question'])
        print(syn_response)

print(f"""Over-refusal Lingo: {syn_refusal_lingo}/{syn_lingo}""")
print(f"""Over-refusal Reworded: {syn_refusal_reworded}/{syn_reworded}""")
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from pathlib import Path

load_dotenv(".env")
random.seed(97)
CHUNKS_PATH = Path("data") / "chunks.json"
SYNTHETIC_QUESTIONS_PATH = Path("eval") / "synthetic_questions.json"
SYNTHETIC_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
INSTRUCTIONS = """- You receive a passage from a quantum physics paper and write one and only one question related to the passage, where the answer lies in the source passage.
- The question has to be AUTONOMOUS: No "in this passage" or "according to the authors",
- The question MUST be understandable WITHOUT the passage: name the model, experiment, or phenomenon it is about.
- If the passage doesn't have any scientific content, if it contains thanks or data on authors, answer "SKIP"
- OUTPUT THE QUESTION, OR SAY SKIP.
"""

if SYNTHETIC_QUESTIONS_PATH.exists():
    print("Already generated, delete the file to regenerate")

else:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as j:
        chunks = json.load(j)

    client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

    candidates = []
    for chunk in chunks:
        if len(chunk['text'].split())>=50:
            candidates.append(chunk)

    sampled = random.sample(candidates, 30)
    print(len(candidates))

    questions = []
    skip_counter = 0
    for i, chunk in enumerate(sampled, start=0):
        if i%2 == 0:
            question_type = "lingo"
            instruction_type = "use key technical terms from the following passage to generate the question"
        else:
            question_type = "reworded"
            instruction_type = "do not use key technical terms from the following passage to generate the question, just ask the same information with simple words or synonymes"
        
        prompt = f"{instruction_type}:\n\n{chunk['text']}"

        response = client.responses.create(
            model = "gpt-5.6-luna",
            instructions = INSTRUCTIONS,
            input = prompt
        )

        cleaned_response = response.output_text.strip()
        
        if cleaned_response == "SKIP":
            skip_counter+=1
            continue
        questions.append({'question':cleaned_response, 'type':question_type, 'expected':[{'paper_id':chunk['paper_id'], 'section':chunk['section']}]})

    with open(SYNTHETIC_QUESTIONS_PATH, "w", encoding="utf-8") as j:
        json.dump(questions, j, ensure_ascii=False, indent=2)

    print(len(questions))
    print(skip_counter)
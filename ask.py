from dotenv import load_dotenv
from openai import OpenAI
import os
from retrieve import search_hybrid, embeddings, bm25, chunks

load_dotenv(".env")

INSTRUCTIONS = """You are an assistant that only answers based on the context, do not use your own knowledge.
if the question doesn't have the answer in the context answer exactly: "I can't find the answer in the corpus",
your answer has to mention the source in the following format : AFFIRMATION
[N] (XXXX.YYYY, RELEVANT SECTION TITLE)"""


def build_context(results):
    """Format the retrieved chunks as a numbered context for the LLM.
    Each chunk becomes a block labeled [N] (paper_id, section), followed
    by its text, so the model can cite its sources by number.
    Blocks are separated by a blank line.
    """
    context = ""
    for i in range(len(results)):
        context += f"[{i+1}] ({results[i]['paper_id']}, {results[i]['section']})\n{results[i]['text']}\n\n"
    return context


client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
question = "what is the CHSH inequality?"
#question = "what is photosynthesis?"
results = search_hybrid(question, embeddings ,bm25, chunks)
context = build_context(results)
prompt = "Context:\n"+ context + "\n\n" + "Question:\n" + question

response = client.responses.create(
    model = "gpt-5.6-luna",
    instructions = INSTRUCTIONS,
    input = prompt
)

print(response.output_text)
from dotenv import load_dotenv
from openai import OpenAI
import os
from retrieve import search_hybrid, embeddings, bm25, chunks

load_dotenv(".env")
client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

REFUSAL = "I can't find the answer in the corpus"

INSTRUCTIONS = f"""You are an assistant that only answers based on the context, do not use your own knowledge.
if the question doesn't have the answer in the context answer exactly: "{REFUSAL}",
Cite the source number in brackets after each claim, for example:
"Quantum mechanics can reach 2√2 [5]."
End with the list of cited sources, one per line: [N] (paper_id, section).
"""

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

def answer(question):
    """Answer a question using only the papers in the corpus.
    Retrieves the top 5 chunks with hybrid search, sends them as a numbered
    context to gpt-5.6-luna, and returns its answer as a string. The answer
    cites its sources as [N], or is exactly REFUSAL when the context does
    not contain the answer.
    """
    results = search_hybrid(question, embeddings ,bm25, chunks)
    context = build_context(results)
    prompt = "Context:\n"+ context + "\n\n" + "Question:\n" + question
    response = client.responses.create(
        model = "gpt-5.6-luna",
        instructions = INSTRUCTIONS,
        input = prompt
    )
    return (response.output_text)

if __name__ == "__main__":
    #question_1 = "what is the CHSH inequality?"
    #response_1 = answer(question_1)
    
    question_2 = "What is the detection loophole in Bell experiments?"
    print(question_2)
    response_2 = answer(question_2)
    print(response_2)
    print("\n")
    #question_3 = "what is photosynthesis?"
    #response_3 = answer(question_3)
    print(80*"=")
    print("\n")
    question_4 = "What distance did the Micius satellite reach for quantum key distribution?"
    response_4 = answer(question_4)
    print(question_4)
    print(response_4)
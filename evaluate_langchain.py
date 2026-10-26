from metrics import is_hit, is_paper_hit
from rag_langchain import retriever, bm25_retriever_tokenize, ensemble
import json
from pathlib import Path
SYN_QUESTIONS_PATH = Path("eval")/"synthetic_questions.json"

def to_metadatas(documents):
    """Return the metadata dict of each document, so results can be checked with is_hit."""
    metadata_list = []
    for doc in documents:
        metadata_list.append(doc.metadata)
    return metadata_list

if __name__ == "__main__":

    with open(SYN_QUESTIONS_PATH, "r", encoding="utf-8") as j:
        syn_questions = json.load(j)

    hits_vector = 0
    hits_bm25_tokenize = 0
    hits_ensemble = 0

    for question in syn_questions:
        relevant_vec = retriever.invoke(question['question'])
        relevant_bm25_tokenize = bm25_retriever_tokenize.invoke(question['question'])
        relevant_ensemble = ensemble.invoke(question['question'])[:5]

        metadatas_vec = to_metadatas(relevant_vec)
        metadatas_bm25_tokenize = to_metadatas(relevant_bm25_tokenize)
        metadatas_ensemble = to_metadatas(relevant_ensemble)

        if is_hit(metadatas_vec, question['expected']):
            hits_vector += 1
        if is_hit(metadatas_bm25_tokenize, question['expected']):
            hits_bm25_tokenize += 1
        if is_hit(metadatas_ensemble, question['expected']):
            hits_ensemble += 1
    
    print("VECTOR")
    print(f"Hit@5 score: {hits_vector}/{len(syn_questions)} - {100*hits_vector/len(syn_questions):.1f}%")
    print("BM25 TOKENIZED")
    print(f"Hit@5 score: {hits_bm25_tokenize}/{len(syn_questions)} - {100*hits_bm25_tokenize/len(syn_questions):.1f}%")
    print("ENSEMBLE")
    print(f"Hit@5 score: {hits_ensemble}/{len(syn_questions)} - {100*hits_ensemble/len(syn_questions):.1f}%")
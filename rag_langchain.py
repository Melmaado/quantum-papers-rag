from pathlib import Path
from langchain_core.documents import Document
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from text_utils import tokenize
from langchain_classic.retrievers import EnsembleRetriever

CHUNKS_PATH = Path("data")/"chunks.json"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

with open(CHUNKS_PATH, "r", encoding="utf-8") as c:
    chunks = json.load(c)

docs = []
for chunk in chunks:
    text = chunk['text']
    data = {'paper_id' : chunk['paper_id'], 'section' : chunk['section']}
    doc = Document(page_content=text, metadata=data)
    docs.append(doc)

encoding = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(model_name=MODEL, encode_kwargs=encoding)

vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
searching = {'k':5}
retriever = vector_store.as_retriever(search_kwargs=searching)

bm25_retriever_default = BM25Retriever.from_documents(documents=docs, k=5)

bm25_retriever_tokenize = BM25Retriever.from_documents(documents=docs, preprocess_func=tokenize, k=5)

searching_20 = {'k':20}
retriever_20 = vector_store.as_retriever(search_kwargs=searching_20)
bm25_retriever_tokenize_20 = bm25_retriever_tokenize.model_copy(update={"k": 20})
ensemble = EnsembleRetriever(retrievers=[retriever_20, bm25_retriever_tokenize_20], weights=[1,1])

def print_results(documents):
    """Print the rank, paper_id and section of each retrieved document, best first."""
    for n, document in enumerate(documents,1):
        print(n)
        print(f"relevant paper_id: {document.metadata['paper_id']}")
        print(f"relevant section: {document.metadata['section']}")
        print("\n")

if __name__ == "__main__":

    print("VECTOR")
    relevant = retriever.invoke("What is the CHSH inequality?")
    print_results(relevant)

    print("BM25 DEFAULT")
    relevant_bm25 = bm25_retriever_default.invoke("What is the CHSH inequality?")
    print_results(relevant_bm25)
    
    print("BM25 WITH TOKENIZER")
    relevant_bm25_tokenize = bm25_retriever_tokenize.invoke("What is the CHSH inequality?")
    print_results(relevant_bm25_tokenize)

    print("ENSEMBLE RETRIEVER")
    full_ensemble = ensemble.invoke("What is the CHSH inequality?")
    relevant_ensemble = full_ensemble[:5]
    print_results(relevant_ensemble)
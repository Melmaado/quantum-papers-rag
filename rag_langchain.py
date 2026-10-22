from pathlib import Path
from langchain_core.documents import Document
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from retrieve import tokenize
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

relevant = retriever.invoke("What is the CHSH inequality?")
print("VECTOR")
for n, document in enumerate(relevant,1):
    print(n)
    print(f"relevant paper_id: {document.metadata['paper_id']}")
    print(f"relevant section: {document.metadata['section']}")
    print("\n")

bm25_retriever_default = BM25Retriever.from_documents(documents=docs, k=5)
relevant_bm25 = bm25_retriever_default.invoke("What is the CHSH inequality?")
print("BM25 DEFAULT")

for n, document in enumerate(relevant_bm25,1):
    print(n)
    print(f"relevant paper_id: {document.metadata['paper_id']}")
    print(f"relevant section: {document.metadata['section']}")
    print("\n")

bm25_retriever_tokenize = BM25Retriever.from_documents(documents=docs, preprocess_func=tokenize, k=5)
relevant_bm25_tokenize = bm25_retriever_tokenize.invoke("What is the CHSH inequality?")
print("BM25 WITH TOKENIZER")

for n, document in enumerate(relevant_bm25_tokenize,1):
    print(n)
    print(f"relevant paper_id: {document.metadata['paper_id']}")
    print(f"relevant section: {document.metadata['section']}")
    print("\n")
from typing import TypedDict
from langchain_core.documents import Document
from rag_langchain import ensemble, print_results

class AgentState(TypedDict):
    question: str
    search_query: str
    documents: list[Document]
    is_relevant: bool
    attempts: int
    answer: str


def retrieve(state: AgentState):
    docs = ensemble.invoke(state['search_query'])[:5]
    return {'documents': docs}

if __name__ == "__main__":
    print_results(retrieve({'search_query': "What is the CHSH inequality?"})['documents'])
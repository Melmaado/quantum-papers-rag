from typing import TypedDict
from langchain_core.documents import Document
from rag_langchain import ensemble, print_results
from prompts import GRADE_INSTRUCTIONS
from ask_langchain import llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", GRADE_INSTRUCTIONS),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

grade_chain = grade_prompt | llm | StrOutputParser()

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

def grade(state: AgentState):
    question = state['question']
    docs = state['documents']
    context = format_docs(docs)
    verdict = grade_chain.invoke({'context': context,'question': question}).strip().lower()
    return {'is_relevant': verdict.startswith('yes')}


if __name__ == "__main__":
    n=0
    while n<5:
        qs = ["What is the CHSH inequality?", "What distance did the Micius satellite reach for quantum key distribution?", "For the two-party measurement model involving Alice and Bob, what family of numerical procedures can place upper limits on the probabilities that quantum systems can produce?"]
        for q in qs:
            docs = retrieve({'search_query': q})['documents']
            print(grade({'question': q, 'documents': docs}))
        n+=1

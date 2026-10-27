from prompts import REFUSAL, INSTRUCTIONS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rag_langchain import ensemble
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(".env")
LLM_MODEL = "gpt-5.6-luna"

def format_docs(documents):
    """Format the top 5 documents as a numbered context: [N] (paper_id, section), then the text."""
    top_documents = documents[:5]
    context = ""
    for n, doc in enumerate(top_documents,1):
        context += f"[{n}] ({doc.metadata['paper_id']}, {doc.metadata['section']})\n{doc.page_content}\n\n"
    return context

llm = ChatOpenAI(model=LLM_MODEL)
prompt = ChatPromptTemplate.from_messages([
    ("system", INSTRUCTIONS),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ])

chain = (
    {'context': ensemble | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("QUESTION 1")
    q1 = "What is the detection loophole in Bell experiments?"
    print(q1)
    a1 = chain.invoke(q1)
    print(a1)
    print("")
    print(80*"=")
    print("")
    print("QUESTION 2")
    q2 = "What distance did the Micius satellite reach for quantum key distribution?"
    print(q2)
    a2 = chain.invoke(q2)
    print(a2)
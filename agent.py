from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.embeddings import OllamaEmbeddings
import weaviate
import os

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "llama3"

def get_llm():
    return ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

def get_retriever():
    client = weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "localhost"),
        port=8080,
        grpc_port=50051,
    )
    embeddings = OllamaEmbeddings(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="Document",
        text_key="text",
        embedding=embeddings,
    )
    return vectorstore.as_retriever()

# --- Task 3: Query Decomposition ---
decomposition_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries better suited for retrieving relevant documents.
Output strictly a list of questions separated by newlines.

Original question: {question}""")

def decompose_query(question: str) -> List[str]:
    llm = get_llm()
    chain = decomposition_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    return response.strip().split("\n")

# --- Answer Synthesis ---
synthesis_prompt = ChatPromptTemplate.from_template("""Here is the context retrieved for the question:
{context}

Answer the following question using the context provided: {question}
""")

def run_agent(question: str):
    llm = get_llm()
    retriever = get_retriever()
    
    # 1. Decompose
    sub_questions = decompose_query(question)
    print(f"Sub-questions: {sub_questions}")
    
    # 2. Retrieve & Aggregate
    aggregated_context = ""
    for sub_q in sub_questions:
        docs = retriever.invoke(sub_q)
        for doc in docs:
            aggregated_context += doc.page_content + "\n\n"
            
    # 3. Synthesize
    chain = (
        {"context": lambda x: aggregated_context, "question": RunnablePassthrough()}
        | synthesis_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)

if __name__ == "__main__":
    # Test
    try:
        print(run_agent("What are the evaluation metrics?"))
    except Exception as e:
        print(f"Error running agent: {e}")

import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness
)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from agent import run_agent, get_retriever

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "llama3"

print(f"Connecting to Ollama at {OLLAMA_BASE_URL} with model {MODEL_NAME}")

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

# Define Test Data
questions = [
    "What metrics should be used for evaluation?",
    "Which vector store is preferred?",
    "What is the preferred framework due to agentic capabilities?"
]

ground_truths = [
    ["Retrieval accuracy, Retrieval precision, Contextual accuracy, Contextual precision"],
    ["Weaviate"],
    ["LangGraph or LangChain"]
]

print("Running RAG generation...")
answers = []
contexts = []

retriever = get_retriever()

for q in questions:
    try:
        print(f"Processing: {q}")
        # Retrieval
        docs = retriever.invoke(q)
        ctx = [d.page_content for d in docs]
        contexts.append(ctx)
        
        # Generation
        ans = run_agent(q)
        answers.append(ans)
    except Exception as e:
        print(f"Error processing {q}: {e}")
        answers.append("Error")
        contexts.append([])

# Prepare Dataset
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

print("Starting Ragas Evaluation...")
try:
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=llm,
        embeddings=embeddings
    )

    print("\n=== Evaluation Results ===")
    print(results)
    
    # Save results
    df = results.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")
    
except Exception as e:
    print(f"Evaluation Failed: {e}")

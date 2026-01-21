import os
from typing import List
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings # Or local embeddings
from langchain_community.embeddings import OllamaEmbeddings
import weaviate
from weaviate.classes.init import Auth

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_weaviate_client():
    return weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "localhost"),
        port=8080,
        grpc_port=50051,
    )

def ingest_pdf(file_path: str):
    print(f"Ingesting {file_path}...")
    
    # Extract text and tables
    loader = UnstructuredPDFLoader(
        file_path,
        mode="elements",
        strategy="hi_res", # Extracts tables and images metadata
    )
    docs = loader.load()
    
    # Clean metadata (remove complex objects like coordinates which confuse Weaviate)
    for doc in docs:
        if 'coordinates' in doc.metadata:
            del doc.metadata['coordinates']
        if 'points' in doc.metadata:
            del doc.metadata['points']
            
    # Initialize Weaviate Vector Store
    # We will use Ollama for embeddings
    embeddings = OllamaEmbeddings(
        model="llama3", # Ensure this model is pulled in Ollama
        base_url=OLLAMA_BASE_URL
    )
    
    client = get_weaviate_client()
    
    # Index documents
    try:
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name="Document",
            text_key="text",
            embedding=embeddings,
        )
        
        vectorstore.add_documents(docs)
        print(f"Successfully ingested {len(docs)} chunks from {file_path}")
        
    finally:
        client.close()

if __name__ == "__main__":
    # Example usage
    pdf_path = "Debyez AI intern Assessment Steps.pdf"
    if os.path.exists(pdf_path):
        ingest_pdf(pdf_path)
    else:
        print(f"File {pdf_path} not found.")

# Architecture and Evaluation Report

## 1. Architecture Diagram & Data Flow

### Architecture Component
The application is designed as a modular, containerized RAG (Retrieval-Augmented Generation) system.

```mermaid
graph TD
    Client[Client (API/UI)] -->|HTTP POST /query| API[FastAPI Server]
    Client -->|HTTP POST /ingest| API
    
    subgraph "Application Container"
        API -->|Parse PDF| Ingestion[Ingestion Module]
        API -->|Decompose & Synthesize| Agent[RAG Agent]
    end
    
    subgraph "Infrastructure"
        Ingestion -->|Embed & Store| Weaviate[(Weaviate Vector Store)]
        Agent -->|Retrieve Context| Weaviate
        Agent -->|Generate/Embed| Ollama[Local LLM (Ollama)]
        Ingestion -->|Embed| Ollama
    end
```

### Data Flow Summary
1.  **Ingestion**: 
    - PDF documents are uploaded via the `/ingest` endpoint.
    - Providing multi-modal capability, `unstructured` (or `pypdf`/`pdf2image`) extracts text, tables, and images.
    - Content is chunked and embedded using a local LLM (e.g., Llama 3) via Ollama.
    - Embeddings and metadata are stored in **Weaviate**.

2.  **Retrieval & Generation (Task 1 & 3)**:
    - **Query Decomposition**: The user's query is analyzed by the Agent and decomposed into sub-questions to capture different aspects of the request.
    - **Retrieval**: Each sub-question queries the Weaviate vector store to retrieve relevant context chunks.
    - **Aggregation**: Results from all sub-queries are aggregated.
    - **Synthesis**: The LLM synthesizes a final answer based on the aggregated context and the original query.

## 2. Evaluation Report (Task 2)

The system includes an evaluation pipeline (`evaluation.ipynb`) utilizing the **Ragas** framework to assess performance.

### Metrics Evaluated
-   **Context Precision**: Measures if the retrieved chunks are relevant to the query.
-   **Context Recall**: Measures if the retrieved chunks contain the ground truth answer.
-   **Faithfulness**: Measures if the generated answer is factually consistent with the retrieved context.
-   **Answer Relevancy**: Measures how relevant the generated answer is to the query.

### Execution
To run the evaluation:
1.  Ensure the stack is running.
2.  Install dependencies: `pip install ragas datasets`.
3.  Run the `evaluation.ipynb` notebook.
*(Note: Requires a ground truth dataset for accurate scoring.)*

## 3. Milestones Achieved
-   [x] **Task 1 (App Development)**: Created FastAPI server with Ingestion and Retrieval endpoints. Configured Weaviate and Ollama.
-   [x] **Task 2 (Evaluation)**: Implemented `evaluation.ipynb` using Ragas.
-   [x] **Task 3 (Enhancements)**: Implemented Agentic RAG with Query Decomposition and Answer Synthesis in `agent.py`.
-   [x] **Task 4 (Dockerization)**: Fully containerized application with `Dockerfile` and `docker-compose.yml`.

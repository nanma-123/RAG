import requests
import time
import os

BASE_URL = "http://localhost:8000"

def wait_for_service(retries=30, delay=5):
    print("Waiting for RAG App to be ready...")
    for i in range(retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Service is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(f"Waiting... ({i+1}/{retries})")
        time.sleep(delay)
    return False

def test_ingestion():
    file_path = "Debyez AI intern Assessment Steps.pdf"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return False
        
    print(f"Testing ingestion of {file_path}...")
    with open(file_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(f"{BASE_URL}/ingest", files=files)
            if response.status_code == 200:
                print("Ingestion Success:", response.json())
                return True
            else:
                print("Ingestion Failed:", response.text)
                return False
        except Exception as e:
            print(f"Ingestion Error: {e}")
            return False

def test_query():
    query = "What are the evaluation metrics?"
    print(f"Testing query: '{query}'...")
    try:
        response = requests.post(f"{BASE_URL}/query", json={"query": query})
        if response.status_code == 200:
            print("Query Success:", response.json())
            return True
        else:
            print("Query Failed:", response.text)
            return False
    except Exception as e:
        print(f"Query Error: {e}")
        return False

if __name__ == "__main__":
    if wait_for_service():
        ingest_ok = test_ingestion()
        if ingest_ok:
            test_query()
        else:
            print("Skipping query test due to ingestion failure.")
    else:
        print("Service failed to start.")

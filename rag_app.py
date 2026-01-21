from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from agent import run_agent
from ingestion import ingest_pdf

app = FastAPI(title="Debyez RAG System")

class QueryRequest(BaseModel):
    query: str

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # Ingest the file
        ingest_pdf(file_location)
        
        # Cleanup
        os.remove(file_location)
        
        return {"message": f"Successfully ingested {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_system(request: QueryRequest):
    try:
        result = run_agent(request.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

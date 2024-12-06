from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG Chatbot Service")

class HealthCheck(BaseModel):
    status: str
    version: str

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return HealthCheck(status="healthy", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "RAG Chatbot Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090) 
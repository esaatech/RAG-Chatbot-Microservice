from fastapi import FastAPI, HTTPException, UploadFile, File,Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import uuid
from dotenv import load_dotenv
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
import json
from datetime import datetime
from fastapi.openapi.docs import get_swagger_ui_html

# Load environment variables
load_dotenv()
#assert os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is not None, "Google credentials not found"
#assert os.getenv('OPENAI_API_KEY') is not None, "OpenAI API key not found"  
from app.services.chatbot import ChatbotService
# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")
app = FastAPI(
    title="RAG Chatbot API",
    description="""
    A RAG (Retrieval-Augmented Generation) Chatbot API that processes documents, 
    handles vector storage, and provides conversational responses.
    
    Key Features:
    - Document processing and vectorization
    - Configurable prompt handling
    - Caching system
    - Chat history management
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot service
chatbot_service = ChatbotService()

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    prompt_config: Optional[str] = Form(None)
):
    """
        Upload and process a new document.

        Parameters:
        - file: Document file (PDF, TXT, DOCX, CSV)
        - prompt_config: Optional configuration for response generation
            {
                "company_name": str,
                "agent_role": str,
                "response_style": str,
                "fallback_message": str,
                "tone": str
            }

        Returns:
        {
            "key": str,  # Unique document identifier
            "message": str,
            "config": Dict  # Applied configuration
        }

        Example:
        ```python
        files = {'file': open('document.pdf', 'rb')}
        config = {
            'company_name': 'TestCo',
            'tone': 'professional'
        }
        response = requests.post('/documents/upload', files=files, json={'prompt_config': config})
        ```
        """
    try:
        document_key = str(uuid.uuid4())
        
        # Parse prompt_config from JSON string if provided
        config_dict = json.loads(prompt_config) if prompt_config else None
        
        success = chatbot_service.process_document(
            file=file.file,
            key=document_key,
            filename=file.filename,
            prompt_config=config_dict
        )
        
        if success:
            return {
                "key": document_key,
                "prompt_config": config_dict,
                "message": "Document processed successfully"
            }
        raise HTTPException(status_code=500, detail="Failed to process document")
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in prompt_config")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/documents/{document_key}/query")
async def query_document(
    document_key: str, 
    query: str = Query(None),  # From URL query parameter
    body: dict = Body(None)    # From request body
):
    """Query a specific document"""
    try:
        # Use query from URL parameter or body
        query_text = query or body.get("query")
        if not query_text:
            raise HTTPException(status_code=400, detail="Query is required")
            
        response = chatbot_service.query_document(
            key=document_key,
            query=query_text
        )
        return {"response": response}
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document with key {document_key} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_key}")
async def delete_document(document_key: str):
    """Delete a document"""
    try:
        success = chatbot_service.delete_vectorstore(key=document_key)
        if success:
            return {"message": "Document deleted successfully"}
        raise HTTPException(status_code=404, detail=f"Document with key {document_key} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# app/main.py
@app.put("/documents/{document_key}/update")
async def update_document(
    document_key: str,
    file: UploadFile = File(...),
    prompt_config: Optional[str] = Form(None)  # Changed to str to accept JSON string
    ):   

    """Update an existing document with optional prompt configuration"""
    try:
        new_key = str(uuid.uuid4())
        
        # Parse prompt_config from JSON string if provided
        config_dict = json.loads(prompt_config) if prompt_config else None
        
        success = chatbot_service.update_document(
            old_key=document_key,
            new_key=new_key,
            file=file.file,
            filename=file.filename,
            prompt_config=config_dict
        )
        
        if success:
            return {
                "old_key": document_key,
                "new_key": new_key,
                "prompt_config": config_dict,
                "message": "Document updated successfully"
            }
        raise HTTPException(status_code=500, detail="Failed to update document")
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in prompt_config")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document with key {document_key} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/documents/{key}/update-config")
async def update_document_config(
    key: str,
    prompt_config: Dict = Body(..., description="New prompt configuration")
) -> Dict:
    """Update only the document's configuration."""
    try:
        # Get existing metadata
        metadata = chatbot_service._get_document_metadata(key)
        
        # Update the config correctly (fix the nesting issue)
        if "prompt_config" in prompt_config:
            # Extract the inner config
            new_config = prompt_config["prompt_config"]
        else:
            new_config = prompt_config

        # Create updated config
        updated_config = {
            **chatbot_service.DEFAULT_PROMPT_CONFIG,
            **new_config
        }
        
        # Update metadata with the correct structure
        metadata["prompt_config"] = updated_config
        metadata["updated_at"] = datetime.utcnow().isoformat()
        
        # Save updated metadata
        chatbot_service._save_document_metadata(key, metadata)
        
        return {
            "message": "Configuration updated successfully",
            "key": key,
            "config": metadata["prompt_config"]  # Return only the prompt config
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating configuration: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
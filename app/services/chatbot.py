# app/services/chatbot.py
import os
import uuid
import shutil
import time
import tempfile
import json
from datetime import datetime
from typing import Dict, BinaryIO, List, Optional
from collections import defaultdict
from pathlib import Path
from google.cloud import storage

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import HumanMessage, SystemMessage

# Document Loaders
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    Docx2txtLoader,
    CSVLoader
)

# Local imports
from .cache import ConfigCache

class ChatbotService:
    """Service for handling document processing, vectorization, and querying with GCS storage."""
    
    DEFAULT_PROMPT_CONFIG = {
        "company_name": "Esaa",
        "agent_role": "customer support agent",
        "response_style": "Ensure to always return concise and shortest possible answer as defined in the document.",
        "fallback_message": "I will transfer you to a superior for better assistance.",
        "tone": "professional and concise"
    }

    def __init__(self):
        """Initialize the ChatbotService with GCS integration and caching."""
        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4")
        
        # Initialize state storage
        self.chat_histories = defaultdict(list)
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket_name = "rag-chatbot-vectors-esaasolution"
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        # Initialize config cache
        self.config_cache = ConfigCache(capacity=100, expiration_time=3600)
        
        # Set up temporary directory for processing
        self.temp_dir = Path(tempfile.gettempdir()) / "rag-chatbot"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _get_document_metadata(self, key: str) -> dict:
        """Get document metadata from cache or GCS."""
        # Try to get from cache first
        cached_metadata = self.config_cache.get(key)
        if cached_metadata:
            return cached_metadata

        # If not in cache, get from GCS
        try:
            blob = self.bucket.blob(f"vectorstore/{key}/metadata.json")
            metadata = json.loads(blob.download_as_string())
            # Store in cache for future use
            self.config_cache.put(key, metadata)
            return metadata
        except Exception as e:
            print(f"Error getting metadata from GCS: {e}")
            return {"prompt_config": self.DEFAULT_PROMPT_CONFIG}

    def _save_document_metadata(self, key: str, metadata: dict) -> None:
        """Save document metadata to both cache and GCS."""
        try:
            # Save to GCS
            blob = self.bucket.blob(f"vectorstore/{key}/metadata.json")
            blob.upload_from_string(
                json.dumps(metadata),
                content_type='application/json'
            )
            # Update cache
            self.config_cache.put(key, metadata)
        except Exception as e:
            print(f"Error saving metadata: {e}")
            raise

    def _upload_to_gcs(self, source_path: Path, key: str):
        """Upload files to GCS bucket."""
        prefix = f"vectorstore/{key}"
        for file_path in source_path.glob('**/*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(source_path)
                blob = self.bucket.blob(f"{prefix}/{relative_path}")
                blob.upload_from_filename(str(file_path))

    def _download_from_gcs(self, key: str, target_path: Path):
        """Download files from GCS bucket."""
        prefix = f"vectorstore/{key}"
        blobs = self.bucket.list_blobs(prefix=prefix)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for blob in blobs:
            relative_path = Path(blob.name).relative_to(prefix)
            local_path = target_path / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))

    def _delete_from_gcs(self, key: str):
        """Delete files from GCS bucket and clear cache."""
        prefix = f"vectorstore/{key}"
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            blob.delete()
        # Clear from cache
        self.config_cache.remove(key)

    # ... (keep existing helper methods like _save_temp_file, _cleanup_temp_file, etc.)

    def process_document(self, file: BinaryIO, key: str, filename: str, prompt_config: Optional[Dict] = None) -> bool:
        """Process and vectorize a document, storing in GCS with metadata."""
        temp_vectorstore_path = self.temp_dir / f"vectorstore_{key}"
        try:
            temp_file_path = self._save_temp_file(file, filename)
            
            try:
                # Prepare metadata
                metadata = {
                    "prompt_config": {**self.DEFAULT_PROMPT_CONFIG, **(prompt_config or {})},
                    "filename": filename,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }

                # Load and split document
                documents = self.load_and_split_document(str(temp_file_path), filename)
                if not documents:
                    raise ValueError(f"No content extracted from file: {filename}")

                # Vectorize documents locally first
                self.vectorize_documents(documents, str(temp_vectorstore_path))
                
                # Upload to GCS
                self._upload_to_gcs(temp_vectorstore_path, key)
                
                # Save metadata
                self._save_document_metadata(key, metadata)
                
                return True

            finally:
                self._cleanup_temp_file(temp_file_path)
                if temp_vectorstore_path.exists():
                    shutil.rmtree(str(temp_vectorstore_path))

        except Exception as e:
            print(f"Error processing document: {e}")
            self._delete_from_gcs(key)
            raise

    def query_document(self, key: str, query: str) -> str:
        """Query the document using cached config when available."""
        temp_vectorstore_path = self.temp_dir / f"vectorstore_{key}"
        try:
            # Get metadata from cache or GCS
            metadata = self._get_document_metadata(key)
            config = metadata.get("prompt_config", self.DEFAULT_PROMPT_CONFIG)
            
            # Download vectors from GCS
            self._download_from_gcs(key, temp_vectorstore_path)
            
            try:
                db = Chroma(
                    persist_directory=str(temp_vectorstore_path), 
                    embedding_function=self.embeddings
                )
                
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                
                history_aware_retriever = create_history_aware_retriever(
                    self.llm,
                    retriever,
                    self._create_contextualize_prompt()
                )
                
                question_answer_chain = create_stuff_documents_chain(
                    self.llm,
                    self._create_qa_prompt(config)
                )
                
                rag_chain = create_retrieval_chain(
                    history_aware_retriever, 
                    question_answer_chain
                )

                chat_history = self.chat_histories[key]
                result = rag_chain.invoke({
                    "input": query, 
                    "chat_history": chat_history
                })

                chat_history.append(HumanMessage(content=query))
                chat_history.append(SystemMessage(content=result["answer"]))
                self.chat_histories[key] = chat_history

                return result["answer"]

            finally:
                if temp_vectorstore_path.exists():
                    shutil.rmtree(str(temp_vectorstore_path))

        except Exception as e:
            print(f"Error querying document: {e}")
            raise

    # app/services/chatbot.py
# ... (previous code remains the same until update_document method)

    def update_document(self, old_key: str, new_key: str, file: BinaryIO, filename: str, 
                       prompt_config: Optional[Dict] = None) -> bool:
        """
        Update a document with a new version in GCS.
        
        Args:
            old_key (str): Key of the document to update
            new_key (str): Key for the updated document
            file (BinaryIO): New document file
            filename (str): Name of the file
            prompt_config (Optional[Dict]): New prompt configuration
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Get old metadata before deletion (for timestamp preservation)
            try:
                old_metadata = self._get_document_metadata(old_key)
                created_at = old_metadata.get('created_at')
            except Exception:
                created_at = datetime.utcnow().isoformat()

            # Process new document
            temp_vectorstore_path = self.temp_dir / f"vectorstore_{new_key}"
            temp_file_path = self._save_temp_file(file, filename)
            
            try:
                # Prepare new metadata
                metadata = {
                    "prompt_config": {**self.DEFAULT_PROMPT_CONFIG, **(prompt_config or {})},
                    "filename": filename,
                    "created_at": created_at,  # Preserve original creation time
                    "updated_at": datetime.utcnow().isoformat(),
                    "previous_key": old_key  # Track document history
                }

                # Load and split document
                documents = self.load_and_split_document(str(temp_file_path), filename)
                if not documents:
                    raise ValueError(f"No content extracted from file: {filename}")

                # Vectorize documents locally first
                self.vectorize_documents(documents, str(temp_vectorstore_path))
                
                # Upload to GCS
                self._upload_to_gcs(temp_vectorstore_path, new_key)
                
                # Save new metadata
                self._save_document_metadata(new_key, metadata)
                
                # Delete old document and its cache
                self._delete_document_completely(old_key)
                
                return True

            finally:
                self._cleanup_temp_file(temp_file_path)
                if temp_vectorstore_path.exists():
                    shutil.rmtree(str(temp_vectorstore_path))

        except Exception as e:
            print(f"Error updating document: {e}")
            # Clean up new document if update failed
            self._delete_document_completely(new_key)
            raise

    def _delete_document_completely(self, key: str) -> None:
        """
        Completely delete a document, including GCS storage, cache, and chat history.
        
        Args:
            key (str): Document key to delete
        """
        try:
            # Delete from GCS
            self._delete_from_gcs(key)
            # Clear from cache
            self.config_cache.remove(key)
            # Clear chat history
            self.chat_histories.pop(key, None)
        except Exception as e:
            print(f"Error during complete document deletion: {e}")
            raise

    def delete_vectorstore(self, key: str) -> bool:
        """
        Delete vector store and all associated data.
        
        Args:
            key (str): Document key to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            self._delete_document_completely(key)
            return True
        except Exception as e:
            print(f"Error deleting vectorstore: {e}")
            raise

    def get_document_info(self, key: str) -> Dict:
        """Get information about a document using cached metadata when available."""
        try:
            # Get metadata from cache or GCS
            metadata = self._get_document_metadata(key)
            
            return {
                "key": key,
                "exists": True,
                "metadata": metadata,
                "chat_history_length": len(self.chat_histories[key])
            }
        except Exception as e:
            print(f"Error getting document info: {e}")
            raise
    def clear_chat_history(self, key: str) -> bool:
        """
        Clear chat history for a specific document.
        
        Args:
            key (str): Document key to clear history for
            
        Returns:
            bool: True if history was cleared, False if key didn't exist
        """
        if key in self.chat_histories:
            self.chat_histories[key] = []
            return True
        return False
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict: Dictionary containing cache statistics
        """
        return self.config_cache.get_stats()
    def __del__(self):
        """Cleanup on service destruction."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(str(self.temp_dir))
        except Exception as e:
            print(f"Error during cleanup: {e}")

    
    

    def _save_temp_file(self, file: BinaryIO, filename: str) -> Path:
        """
        Save uploaded file temporarily.
        
        Args:
            file (BinaryIO): File object to save
            filename (str): Original filename
            
        Returns:
            Path: Path to saved temporary file
        """
        file_extension = Path(filename).suffix
        safe_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = self.temp_dir / safe_filename
        
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file, buffer)
        
        return temp_path

    def _cleanup_temp_file(self, file_path: Path):
        """
        Clean up temporary file.
        
        Args:
            file_path (Path): Path to file to clean up
        """
        if file_path.exists():
            file_path.unlink()

    def _create_contextualize_prompt(self) -> ChatPromptTemplate:
        """Create the contextualization prompt template."""
        system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def _create_qa_prompt(self, config: Dict) -> ChatPromptTemplate:
        """
        Create the QA prompt template with configuration.
        
        Args:
            config (Dict): Prompt configuration
            
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        system_prompt = (
            f"You are a {config['agent_role']} for {config['company_name']}. "
            f"Use the provided retrieved context to answer the question. "
            f"{config['response_style']} "
            f"If you don't know the answer, {config['fallback_message']} "
            f"Maintain a {config['tone']} tone."
            "\n\n"
            "{context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def load_and_split_document(self, file_path: str, filename: str) -> List:
        """
        Load and split a document into chunks.
        
        Args:
            file_path (str): Path to document file
            filename (str): Original filename
            
        Returns:
            List: List of document chunks
        """
        file_extension = Path(filename).suffix.lower()
        loader_map = {
            ".txt": TextLoader,
            ".pdf": PDFMinerLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader
        }
        
        if file_extension not in loader_map:
            raise ValueError(f"Unsupported file type: {file_extension}")

        loader = loader_map[file_extension](file_path)
        documents = loader.load()

        if not documents:
            return []

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    def vectorize_documents(self, documents: List, persist_directory: str) -> Chroma:
        """
        Vectorize documents and store them.
        
        Args:
            documents (List): List of documents to vectorize
            persist_directory (str): Directory to store vectors
            
        Returns:
            Chroma: Vectorstore instance
        """
        if not documents:
            raise ValueError("No documents to vectorize")

        vectorstore = Chroma.from_documents(
            documents,
            self.embeddings,
            persist_directory=persist_directory,
        )
        vectorstore.persist()
        return vectorstore
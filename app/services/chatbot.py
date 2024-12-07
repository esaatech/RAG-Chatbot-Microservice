# app/services/chatbot.py
import os
import uuid
import shutil
import time
import tempfile
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
        """Initialize the ChatbotService with GCS integration."""
        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4")
        
        # Initialize state storage
        self.chat_histories = defaultdict(list)
        self.prompt_configs = {}
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket_name = "rag-chatbot-vectors-esaasolution"
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        # Set up temporary directory for processing
        self.temp_dir = Path(tempfile.gettempdir()) / "rag-chatbot"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

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
        """Delete files from GCS bucket."""
        prefix = f"vectorstore/{key}"
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            blob.delete()

    def _save_temp_file(self, file: BinaryIO, filename: str) -> Path:
        """Save uploaded file temporarily."""
        file_extension = Path(filename).suffix
        safe_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = self.temp_dir / safe_filename
        
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file, buffer)
        
        return temp_path

    def _cleanup_temp_file(self, file_path: Path):
        """Clean up temporary file."""
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
        """Create the QA prompt template with configuration."""
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
        """Load and split a document into chunks."""
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
        """Vectorize documents and store them."""
        if not documents:
            raise ValueError("No documents to vectorize")

        vectorstore = Chroma.from_documents(
            documents,
            self.embeddings,
            persist_directory=persist_directory,
        )
        vectorstore.persist()
        return vectorstore

    def process_document(self, file: BinaryIO, key: str, filename: str, prompt_config: Optional[Dict] = None) -> bool:
        """Process and vectorize a document, storing in GCS."""
        temp_vectorstore_path = self.temp_dir / f"vectorstore_{key}"
        try:
            # Save file temporarily
            temp_file_path = self._save_temp_file(file, filename)
            
            try:
                # Store prompt configuration
                self.prompt_configs[key] = {
                    **self.DEFAULT_PROMPT_CONFIG,
                    **(prompt_config or {})
                }

                # Load and split document
                documents = self.load_and_split_document(str(temp_file_path), filename)
                if not documents:
                    raise ValueError(f"No content extracted from file: {filename}")

                # Vectorize documents locally first
                self.vectorize_documents(documents, str(temp_vectorstore_path))
                
                # Upload to GCS
                self._upload_to_gcs(temp_vectorstore_path, key)
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
        """Query the document from GCS and get a response."""
        temp_vectorstore_path = self.temp_dir / f"vectorstore_{key}"
        try:
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
                    self._create_qa_prompt(self.prompt_configs.get(key, self.DEFAULT_PROMPT_CONFIG))
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

    def delete_vectorstore(self, key: str) -> bool:
        """Delete vector store from GCS."""
        try:
            self._delete_from_gcs(key)
            return True
        except Exception as e:
            print(f"Error deleting from GCS: {e}")
            raise

    def update_document(self, old_key: str, new_key: str, file: BinaryIO, filename: str, 
                       prompt_config: Optional[Dict] = None) -> bool:
        """Update a document with a new version in GCS."""
        try:
            # Process new document
            success = self.process_document(file, new_key, filename, prompt_config)
            if success:
                # Delete old vectorstore from GCS
                self.delete_vectorstore(old_key)
                return True
            return False
        except Exception as e:
            print(f"Error updating document: {e}")
            self.delete_vectorstore(new_key)
            raise

    def get_document_info(self, key: str) -> Dict:
        """Get information about a document."""
        try:
            # Check if document exists in GCS
            prefix = f"vectorstore/{key}"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            exists = len(blobs) > 0

            return {
                "key": key,
                "exists": exists,
                "prompt_config": self.prompt_configs.get(key, self.DEFAULT_PROMPT_CONFIG),
                "chat_history_length": len(self.chat_histories[key])
            }
        except Exception as e:
            print(f"Error getting document info: {e}")
            raise

    def clear_chat_history(self, key: str) -> bool:
        """Clear chat history for a specific document."""
        if key in self.chat_histories:
            self.chat_histories[key] = []
            return True
        return False

    def __del__(self):
        """Cleanup on service destruction."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(str(self.temp_dir))
        except Exception as e:
            print(f"Error during cleanup: {e}")
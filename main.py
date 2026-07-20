"""
Complete RAG Implementation using LangGraph with DOCX Support - FastAPI Version
With AWS Bedrock - No Session Management
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain & LangGraph imports
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# AWS Bedrock imports
from langchain_aws import ChatBedrock, BedrockEmbeddings


# ============================================================================
# AWS BEDROCK HELPERS
# ============================================================================

from llm import get_aws_embeddings, get_aws_llm


# ============================================================================
# HELPER: JSON Serializable Converter
# ============================================================================

def make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration settings for the RAG system"""
    
    # Document processing
    chunk_size: int = 1500
    chunk_overlap: int = 300
    
    # Vector store
    vector_store_path: str = "faiss_index"
    
    # Fixed document path
    document_path: str = "documents/Bhupendra Kumar.docx"
    
    # Retrieval
    retrieval_k: int = 8


# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., description="The question to ask", min_length=1)
    k: Optional[int] = Field(None, description="Number of documents to retrieve", ge=1, le=20)


class QuestionResponse(BaseModel):
    """Response model for question answers"""
    question: str
    answer: str
    context: List[Dict[str, Any]]
    status: str


class InitResponse(BaseModel):
    """Response model for initialization"""
    document_name: str
    chunk_count: int
    status: str
    message: str


# ============================================================================
# STATE DEFINITION (LangGraph)
# ============================================================================

class RAGState(TypedDict):
    """State definition for the RAG workflow"""
    
    question: str
    documents: List[Dict[str, Any]]
    context: str
    answer: str
    messages: Annotated[List, add_messages]


# ============================================================================
# RAG ENGINE
# ============================================================================

class RAGEngine:
    """Main RAG engine using LangGraph workflow with AWS Bedrock"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.llm = None
        self.embeddings = None
        self.workflow = None
        self.document_name = None
        self.chunk_count = 0
        self.is_initialized = False
        self.vector_store_path = None
        
        # Create documents directory if it doesn't exist
        os.makedirs(os.path.dirname(config.document_path), exist_ok=True)
        
        # Initialize AWS Bedrock LLM and Embeddings
        try:
            self.llm = get_aws_llm()
            self.embeddings = get_aws_embeddings()
            print(f"🤖 Using AWS Bedrock LLM: {os.getenv('BEDROCK_LLM_MODEL')}")
            print(f"🔢 Using AWS Bedrock Embeddings: {os.getenv('BEDROCK_EMBEDDING_MODEL')}")
        except Exception as e:
            print(f"⚠️ Could not initialize AWS Bedrock: {e}")
            print("💡 Make sure your AWS credentials are set in .env file")
            raise
    
    def load_document(self) -> List[Dict[str, Any]]:
        """Load the fixed document"""
        doc_path = self.config.document_path
        print(f"📄 Loading document: {doc_path}")
        
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        loader = Docx2txtLoader(doc_path)
        documents = loader.load()
        
        print(f"✅ Loaded {len(documents)} document(s)")
        return documents
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into smaller chunks"""
        print(f"✂️ Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> FAISS:
        """Create vector store from document chunks using AWS Bedrock embeddings"""
        print("🔢 Creating vector store with AWS Bedrock embeddings...")
        
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save vector store
        self.vector_store_path = self.config.vector_store_path
        vector_store.save_local(self.vector_store_path)
        print(f"💾 Vector store saved to: {self.vector_store_path}")
        
        print(f"✅ Vector store created with {len(chunks)} chunks")
        return vector_store
    
    def ingest(self) -> Dict[str, Any]:
        """Complete ingestion pipeline using fixed document"""
        # Load document
        documents = self.load_document()
        
        # Split into chunks
        chunks = self.split_documents(documents)
        self.chunk_count = len(chunks)
        
        # Create vector store
        self.vector_store = self.create_vector_store(chunks)
        
        # Store document name
        self.document_name = os.path.basename(self.config.document_path)
        self.is_initialized = True
        
        return {
            "document_name": self.document_name,
            "chunk_count": self.chunk_count,
            "status": "success"
        }
    
    def load_vector_store(self) -> FAISS:
        """Load existing vector store"""
        if os.path.exists(self.config.vector_store_path):
            print(f"📂 Loading vector store from: {self.config.vector_store_path}")
            self.vector_store = FAISS.load_local(
                self.config.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.is_initialized = True
            print(f"✅ Vector store loaded")
            return self.vector_store
        return None
    
    def retrieve(self, question: str, k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a question"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call ingest() or load_vector_store() first.")
        
        # Use config default if k not provided
        if k is None:
            k = self.config.retrieval_k
        
        print(f"🔍 Retrieving {k} documents for: {question}")
        
        results = self.vector_store.similarity_search_with_score(question, k=k)
        
        documents = []
        for doc, score in results:
            documents.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        print(f"✅ Retrieved {len(documents)} documents")
        return documents
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using AWS Bedrock LLM"""
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        - Answer the question based ONLY on the provided context
        - If the answer is not in the context, say "I don't have enough information to answer this"
        - Be thorough and comprehensive - include ALL relevant information from the context
        - Use bullet points for lists
        - If listing items (like projects, skills, etc.), list ALL of them found in the context
        - Be specific and detailed
        
        Answer:
        """)
        
        chain = prompt_template | self.llm | StrOutputParser()
        
        print(f"💡 Generating answer with AWS Bedrock...")
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        print(f"✅ Answer generated")
        return answer
    
    # ========================================================================
    # LANGGRAPH WORKFLOW NODES
    # ========================================================================
    
    def retrieve_node(self, state: RAGState) -> RAGState:
        """LangGraph node: Retrieve relevant documents"""
        question = state["question"]
        documents = self.retrieve(question)
        
        state["documents"] = documents
        context = "\n\n---\n\n".join([doc["content"] for doc in documents])
        state["context"] = context
        
        return state
    
    def generate_node(self, state: RAGState) -> RAGState:
        """LangGraph node: Generate answer"""
        question = state["question"]
        context = state["context"]
        
        answer = self.generate_answer(question, context)
        state["answer"] = answer
        
        return state
    
    # ========================================================================
    # BUILD WORKFLOW
    # ========================================================================
    
    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        print("🏗️ Building LangGraph workflow...")
        
        workflow = StateGraph(RAGState)
        
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        app = workflow.compile()
        
        print("✅ Workflow built successfully")
        return app
    
    def ask(self, question: str, k: int = None) -> Dict[str, Any]:
        """Ask a question using the RAG system"""
        if not self.is_initialized:
            raise ValueError("RAG system not initialized. Call ingest() or load_vector_store() first.")
        
        if not self.workflow:
            self.workflow = self.build_workflow()
        
        # Use provided k or config default
        if k is None:
            k = self.config.retrieval_k
        
        initial_state = {
            "question": question,
            "documents": [],
            "context": "",
            "answer": "",
            "messages": []
        }
        
        result = self.workflow.invoke(initial_state)
        
        # Convert numpy types to Python native types for JSON serialization
        documents = make_json_serializable(result["documents"])
        
        return {
            "question": question,
            "answer": result["answer"],
            "context": documents,
            "status": "success"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG engine"""
        return {
            "is_initialized": self.is_initialized,
            "document_name": self.document_name,
            "chunk_count": self.chunk_count,
            "vector_store_path": self.vector_store_path,
            "status": "ready" if self.is_initialized else "not_initialized"
        }


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="RAG API with LangGraph and AWS Bedrock",
    description="Complete RAG implementation with document ingestion, Q&A, and AWS Bedrock",
    version="2.0.0"
)

# ============================================================================
# CORS MIDDLEWARE CONFIGURATION
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bkumar0823.space",
        "https://bkumar-portfolio.onrender.com",
        "https://learninglogmanager.onrender.com/openapi.json",
        "https://learninglogmanager.onrender.com",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "all"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
config = RAGConfig()

# Initialize RAG engine (singleton)
rag = RAGEngine(config)

# Try to load existing vector store
try:
    rag.load_vector_store()
except Exception as e:
    print(f"⚠️ Could not load vector store: {e}")


# ============================================================================
# CUSTOM JSON RESPONSE WITH NUMPY HANDLING
# ============================================================================

class NumpyJSONResponse(JSONResponse):
    """Custom JSONResponse that handles numpy types"""
    
    def render(self, content) -> bytes:
        # Convert numpy types to Python native types
        content = make_json_serializable(content)
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    status = rag.get_status()
    return {
        "message": "RAG API with LangGraph and AWS Bedrock",
        "endpoints": [
            "/rag-init - Initialize RAG with fixed document",
            "/rag-status - Check RAG status",
            "/ask - Ask a question (supports custom k parameter)",
            "/health - Health check"
        ],
        "fixed_document": config.document_path,
        "status": status,
        "llm_model": os.getenv("BEDROCK_LLM_MODEL"),
        "embedding_model": os.getenv("BEDROCK_EMBEDDING_MODEL"),
        "version": "2.0.0"
    }


@app.post("/rag-init", response_model=InitResponse)
async def initialize_rag():
    """
    Initialize RAG system with the fixed document
    
    - Uses document at: documents/Bhupendra Kumar.docx
    - Processes and chunks the document
    - Creates vector embeddings using AWS Bedrock
    - Returns initialization status
    """
    global rag
    
    # Check if document exists
    if not os.path.exists(config.document_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document not found at: {config.document_path}. Please place your document there."
        )
    
    try:
        # Ingest document
        result = rag.ingest()
        
        return InitResponse(
            document_name=result["document_name"],
            chunk_count=result["chunk_count"],
            status="success",
            message=f"Document {result['document_name']} ingested successfully with {result['chunk_count']} chunks using AWS Bedrock"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag-status")
async def get_rag_status():
    """
    Get the status of the RAG system
    
    Returns:
    - Initialization status
    - Document name (if initialized)
    - Chunk count (if initialized)
    - Vector store path (if initialized)
    """
    status = rag.get_status()
    return status


@app.post("/ask", response_class=NumpyJSONResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question using the RAG system with AWS Bedrock
    
    Requires an initialized RAG session. Use /rag-init first.
    
    Optional parameter 'k' controls how many documents are retrieved (default: 8)
    Increase k for more comprehensive answers (e.g., listing all projects)
    """
    try:
        result = rag.ask(request.question, k=request.k)
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context": result["context"],
            "status": result["status"]
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    document_exists = os.path.exists(config.document_path)
    status = rag.get_status()
    
    return {
        "status": "healthy",
        "document_path": config.document_path,
        "document_exists": document_exists,
        "vector_store_path": config.vector_store_path,
        "is_initialized": status["is_initialized"],
        "llm_model": os.getenv("BEDROCK_LLM_MODEL"),
        "embedding_model": os.getenv("BEDROCK_EMBEDDING_MODEL"),
        "chunk_size": config.chunk_size,
        "retrieval_k": config.retrieval_k
    }


@app.get("/document-check")
async def check_document():
    """
    Check if the fixed document exists and get its info
    """
    doc_path = config.document_path
    
    if os.path.exists(doc_path):
        file_size = os.path.getsize(doc_path)
        return {
            "exists": True,
            "path": doc_path,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 2),
            "size_mb": round(file_size / (1024 * 1024), 2)
        }
    else:
        return {
            "exists": False,
            "path": doc_path,
            "message": "Document not found. Please place your document at this location."
        }


# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 RAG API WITH LANGGRAPH AND AWS BEDROCK")
    print("="*60 + "\n")
    
    print("📄 Fixed Document Path:")
    print(f"   {config.document_path}")
    
    # Check if document exists
    if os.path.exists(config.document_path):
        print("   ✅ Document found")
        file_size = os.path.getsize(config.document_path)
        print(f"   📊 Size: {round(file_size / 1024, 2)} KB")
    else:
        print("   ❌ Document not found!")
        print(f"   💡 Please place your document at: {config.document_path}")
    
    print(f"\n🔧 AWS Bedrock Configuration:")
    print(f"   LLM Model: {os.getenv('BEDROCK_LLM_MODEL')}")
    print(f"   Embedding Model: {os.getenv('BEDROCK_EMBEDDING_MODEL')}")
    print(f"   AWS Region: {os.getenv('AWS_REGION')}")
    
    print(f"\n🔧 RAG Configuration:")
    print(f"   Chunk Size: {config.chunk_size}")
    print(f"   Chunk Overlap: {config.chunk_overlap}")
    print(f"   Retrieval K: {config.retrieval_k} (documents retrieved per query)")
    
    print("\n📂 Vector Store:")
    print(f"   {config.vector_store_path}")
    
    # Check if vector store exists
    if os.path.exists(config.vector_store_path):
        print("   ✅ Vector store found")
    else:
        print("   ⚠️ No vector store found. Initialize with /rag-init")
    
    print("\n📋 Available Endpoints:")
    print("  POST   /rag-init       - Initialize RAG with fixed document")
    print("  GET    /rag-status     - Check RAG status")
    print("  POST   /ask            - Ask a question (supports custom k)")
    print("  GET    /document-check - Check if document exists")
    print("  GET    /health         - Health check")
    print("\n" + "="*60 + "\n")
    
    print("💡 Quick Start:")
    print("  1. Place your document at: documents/Bhupendra Kumar.docx")
    print("  2. Initialize: POST /rag-init")
    print("  3. Ask question: POST /ask")
    print("  4. For comprehensive answers, increase k:")
    print("     {\"question\": \"List all projects\", \"k\": 10}")
    print("\n💡 Using AWS Bedrock for:")
    print("   - Text Embeddings (vector search)")
    print("   - LLM Generation (answer generation)")
    print("\n💡 Single RAG instance (no session management)")
    print("   - One document loaded at a time")
    print("   - Vector store persists on disk")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


# ============================================================================
# REQUIREMENTS
# ============================================================================

"""
requirements.txt:
-----------------
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
langchain>=0.1.0
langchain-community>=0.1.0
langgraph>=0.0.20
langchain-core>=0.1.0
faiss-cpu>=1.7.4
python-docx>=0.8.11
docx2txt>=0.8
pydantic>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
boto3>=1.34.0
langchain-aws>=0.1.0
"""
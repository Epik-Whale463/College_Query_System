import os
from typing import List
from langchain.document_loaders import (
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader
from typing import Optional, Dict, Any
from dataclasses import dataclass
from langchain.schema import BaseRetriever

load_dotenv()

NVIDIA_API_KEY =  os.getenv("NVIDIA_API_KEY")
QDRANT_URL =  os.getenv("QDRANT_URL")
QDRANT_API_KEY =  os.getenv("QDRANT_API_KEY")
COLLECTION_NAME =  os.getenv("COLLECTION_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Global variables to hold initialized components
embeddings = None
vectorstore = None
qa_chain = None

def load_documents(directory_path: str) -> List:
    try:
        loader = DirectoryLoader(directory_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading documents from '{directory_path}': {str(e)}")
        return []

# Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents: List, chunk_size: int = 750, chunk_overlap: int = 100) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# NVIDIA Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

def get_embeddings(api_key: str, model: str = "snowflake/arctic-embed-l"):
    return NVIDIAEmbeddings(
        nvidia_api_key=api_key,
        model=model
    )

# Qdrant vector store setup
from langchain.vectorstores import Qdrant
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams

def setup_qdrant_store(
    url: str,
    api_key: str,
    collection_name: str,
    embedding_size: int,
    force_recreate: bool = False
) -> qdrant_client.QdrantClient:
    client = qdrant_client.QdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=True
    )
    
    # Check if collection exists
    if force_recreate or not client.collection_exists(collection_name):
        # Delete if exists and force_recreate is True
        if force_recreate and client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=Distance.COSINE
            )
        )

    return client

# Initialize Qdrant vectorstore
def initialize_vectorstore(
    client,
    collection_name: str,
    embeddings,
    texts,
    metadatas=None
):
    return Qdrant.from_texts(
        texts=[doc.page_content for doc in texts],
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        metadatas=metadatas
    )


"""# ChatNVIDIA LLM setup
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import RetrievalQA

def setup_qa_chain(api_key: str, retriever) -> RetrievalQA:
    llm = ChatNVIDIA(
        api_key=api_key,
        model="mistralai/mixtral-8x7b-instruct-v0.1",
        temperature=0.5
    )

    # Define the custom prompt template
    template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a very helpful, clever and friendly assistant for College queries purposes."
            "You ONLY answer questions related to the college based on the provided context."
            "If a question is outside of the scope of the college information in the context, you do not provide an answer."
            "You avoid answering any queries that are not in the {context}."
        ),
        HumanMessagePromptTemplate.from_template(
            "Here is the context:\n{context}\n\nQuestion: {question}"
        )
    ])

    # Create the LLMChain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use "stuff" to pass all documents at once
        retriever=retriever,
        chain_type_kwargs={"prompt": template}
    )

    return qa_chain"""

#LLM setup using GroqCloud
@dataclass
class QAConfig:
    """Configuration class for QA Chain parameters"""
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.4
    max_tokens: Optional[int] = None
    streaming: bool = True
    chain_type: str = "stuff"
    verbose: bool = False
    return_source_documents: bool = True

def create_enhanced_prompt(
    system_instructions: Optional[str] = None,
    custom_formatting: Optional[Dict[str, str]] = None
) -> ChatPromptTemplate:
    """
    Creates an enhanced prompt template with customizable instructions and formatting.
    """
    default_system_template = """You are an advanced AI assistant specialized in answering college-related queries.

Role and Behavior:
- Response about your services should be what you can help them with the CSM Information.
- Maintain a friendly, professional, and welcoming tone
- Provide structured, clear, and concise responses
- Use appropriate formatting for better readability

Response Guidelines:
- Always base answers on the provided {context}
- Never deviate from or add information beyond the {context}
- Strictly answer only college-related questions
- Decline to answer questions outside the provided {context}
- Format responses using:
  * Tables for comparative information
  * Bullet points for lists
  * Markdown for headings and emphasis
  * Code blocks for technical information
  * Numbered lists for sequential information

Additional Instructions:
- Highlight key information using bold or italic text
- Include relevant numerical data when available
- Organize complex information into sections
- Use emoji sparingly but appropriately for engagement
- Provide examples when helpful for understanding
- Avoid jargon and use simple language when possible
Remember: Your primary goal is to provide accurate, helpful, and well-structured information based solely on the provided context."""

    system_template = system_instructions or default_system_template
    default_formatting = {
        "context_prefix": "Context:\n",
        "question_prefix": "Question:",
        "separator": "\n\n"
    }
    
    formatting = {**default_formatting, **(custom_formatting or {})}
    
    human_template = (
        f"{formatting['context_prefix']}{{context}}"
        f"{formatting['separator']}"
        f"{formatting['question_prefix']} {{question}}"
    )

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

def setup_qa_chain(
    retriever: BaseRetriever,
    config: Optional[QAConfig] = None,
    system_instructions: Optional[str] = None,
    custom_formatting: Optional[Dict[str, str]] = None,
    custom_callbacks: Optional[list] = None
) -> RetrievalQA:
    """
    Sets up an enhanced QA chain with configurable parameters and improved prompting.
    
    Args:
        retriever: The retriever instance to use for document retrieval
        config: QAConfig instance for customizing chain parameters
        system_instructions: Custom system instructions for the prompt template
        custom_formatting: Custom formatting for the prompt template
        custom_callbacks: List of callback handlers for the chain
        
    Returns:
        RetrievalQA: Configured QA chain instance
    """
    # Use default config if none provided
    config = config or QAConfig()

    # Initialize Groq LLM with configured parameters
    llm = ChatGroq(
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=config.streaming,
    )

    # Create enhanced prompt template
    prompt = create_enhanced_prompt(
        system_instructions=system_instructions,
        custom_formatting=custom_formatting
    )

    # Create the QA chain with all configurations
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=config.chain_type,
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": config.verbose
        },
        return_source_documents=config.return_source_documents
    )

    # Add custom callbacks if provided
    if custom_callbacks:
        qa_chain.callbacks = custom_callbacks

    return qa_chain

def initialize_rag(file_paths: dict):
    global embeddings, vectorstore, qa_chain

    print("Initializing RAG...")

    # 1. Load documents
    print("Loading documents...")
    documents = load_documents(file_paths)
    if not documents:
        raise Exception("No documents loaded")

    # 2. Split documents
    print("Splitting documents...")
    text_chunks = split_documents(documents)

    # 3. Initialize embeddings
    print("Initializing embeddings...")
    embeddings = get_embeddings(NVIDIA_API_KEY)

    # 4. Setup Qdrant
    print("Setting up Qdrant...")
    client = setup_qdrant_store(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        embedding_size=1024,
        force_recreate=False  # Set to False to avoid accidental recreation
    )

    # 5. Initialize vector store 
    print("Initializing vector store...")
    # Check if the collection already has data
    if client.count(COLLECTION_NAME).count == 0: 
        vectorstore = initialize_vectorstore(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
            texts=text_chunks
        )
    else:
        print("Vector store already exists. Skipping initialization.")
        vectorstore = Qdrant(
            client=client, 
            collection_name=COLLECTION_NAME, 
            embeddings=embeddings
        )

    # 6. Setup retriever
    print("Setting up retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 7. Setup QA chain
    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(retriever)

    print("RAG initialization complete.")

def query_rag(query: str):
    global qa_chain
    if qa_chain is None:
        raise Exception("RAG system not initialized")

    try:
        response = qa_chain.invoke(query)
        return response["result"]
    except Exception as e:
        print(f"Error running query: {str(e)}")
        return "Error processing your query."
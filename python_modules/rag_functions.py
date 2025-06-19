import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from llama_index.core import SimpleDirectoryReader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Qdrant
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

NVIDIA_API_KEY = "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"
QDRANT_URL = "https://7301ad8f-77c3-4cab-8228-e6ab3d6df6b1.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Y71QmeiHy9o7NHt2UmU0CaLvYypJQIHEBiBbE2cjLPU"
COLLECTION_NAME = "my_documents"
GOOGLE_API_KEY = "AIzaSyBCh00jznerGKKK99uQ0smydQfD-bA_y34"
GROQ_API_KEY = "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"
MONGO_URI = "mongodb+srv://sendmail2fa:CSMPASS@csmagent.ch0tn.mongodb.net/?retryWrites=true&w=majority&appName=csmAgent"
MONGO_DB_PASS = "CSMPASS"
PHI_API_KEY = "phi-54SCROX5jEW0_oCzGD002hIKyHKdnuBYYaA9Xhh4NB0"
APP_PASSWORD_GMAIL = "hvjj ulxd gmvw zvlu"
EMAIL_SENDER_ADDRESS = "gcloudsignup@gmail.com"
# Global variables to hold initialized components
embeddings = None
vectorstore = None
qa_chain = None


def load_documents():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    input_d = os.path.join(project_dir, "source_documents")
    
    reader = SimpleDirectoryReader(input_dir=input_d, recursive=True)
    # Convert LlamaIndex documents to LangChain documents
    docs = reader.load_data()
    langchain_docs = [
        Document(page_content=doc.text) 
        for doc in docs
    ]
    return langchain_docs


# Text splitting

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
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
from langchain_community.vectorstores import Qdrant
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
        prefer_grpc=False,
        timeout=120,  # Increased timeout
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
    import time
    
    # Use a very small batch size to avoid timeouts
    batch_size = 5
    
    # If we have a lot of documents, reduce batch size even more
    if len(texts) > 100:
        batch_size = 2
    
    print(f"Uploading {len(texts)} documents with batch size {batch_size}")
    
    try:
        return Qdrant.from_texts(
            texts=[doc.page_content for doc in texts],
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=collection_name,
            metadatas=metadatas,
            batch_size=batch_size,
            timeout=120
        )
    except Exception as e:
        print(f"Error with batch size {batch_size}: {e}")
        print("Trying with batch size 1 and longer timeout...")
        time.sleep(5)  # Wait a bit before retry
        
        try:
            return Qdrant.from_texts(
                texts=[doc.page_content for doc in texts],
                embedding=embeddings,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=collection_name,
                metadatas=metadatas,
                batch_size=1,
                timeout=180
            )
        except Exception as e2:
            print(f"Error with batch size 1: {e2}")
            print("Trying manual upload in very small chunks...")
            return initialize_vectorstore_chunked(
                client, collection_name, embeddings, texts, metadatas
            )

# Fallback method for manual chunked upload
def initialize_vectorstore_chunked(
    client,
    collection_name: str,
    embeddings,
    texts,
    metadatas=None,
    chunk_size=5
):
    import time
    
    # Create empty Qdrant instance first
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    # Upload in very small chunks
    total_docs = len(texts)
    for i in range(0, total_docs, chunk_size):
        chunk_texts = texts[i:i+chunk_size]
        chunk_metadatas = metadatas[i:i+chunk_size] if metadatas else None
        
        print(f"Uploading chunk {i//chunk_size + 1}/{(total_docs + chunk_size - 1)//chunk_size} ({len(chunk_texts)} docs)")
        
        try:
            vectorstore.add_texts(
                texts=[doc.page_content for doc in chunk_texts],
                metadatas=chunk_metadatas
            )
            time.sleep(1)  # Small delay between chunks
        except Exception as e:
            print(f"Error uploading chunk {i//chunk_size + 1}: {e}")
            # Try with even smaller chunk size
            if chunk_size > 1:
                print(f"Retrying with chunk size 1...")
                for j, doc in enumerate(chunk_texts):
                    try:
                        vectorstore.add_texts(
                            texts=[doc.page_content],
                            metadatas=[chunk_metadatas[j]] if chunk_metadatas else None
                        )
                        time.sleep(0.5)
                    except Exception as e3:
                        print(f"Failed to upload document {i+j}: {e3}")
                        continue
            else:
                print(f"Skipping failed chunk {i//chunk_size + 1}")
                continue
    
    print("Chunked upload completed")
    return vectorstore


"""# ChatNVIDIA LLM setup
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import RetrievalQA
from llama_index import SimpleDirectoryReader
from langchain.schema import Document

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
@dataclass
class QAConfig:
    """Configuration class for QA Chain parameters"""
    model_name: str = "meta-llama/llama-prompt-guard-2-22m"  # Updated for Groq
    temperature: float = 0.4
    max_tokens: Optional[int] = None
    streaming: bool = False  # Keep false for stability
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
- Do not provide personal opinions or subjective information
- Do no provide information or data related to what user is not asking for unless the things are very closely related or interlinked.

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

    # Initialize Groq LLM with configured parameters (more stable)
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=config.temperature,
        groq_api_key=GROQ_API_KEY
    )
    
    # Comment out Gemini LLM (having compatibility issues)
    # llm_kwargs = {
    #     "model": "gemini-1.5-flash",
    #     "temperature": config.temperature,
    #     "google_api_key": GOOGLE_API_KEY
    # }
    # 
    # # Only add max_tokens if it's specified
    # if config.max_tokens is not None:
    #     llm_kwargs["max_tokens"] = config.max_tokens
    # 
    # llm = ChatGoogleGenerativeAI(**llm_kwargs)

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

def initialize_rag():
    global embeddings, vectorstore, qa_chain

    print("Initializing RAG...")

    # 1. Load documents
    print("Loading documents...")
    documents = load_documents()
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
    try:
        document_count = client.count(COLLECTION_NAME).count
        print(f"Found {document_count} existing documents in collection")
        
        if document_count == 0:
            print(f"Collection is empty. Starting upload of {len(text_chunks)} documents...")
            vectorstore = initialize_vectorstore(
                client=client,
                collection_name=COLLECTION_NAME,
                embeddings=embeddings,
                texts=text_chunks
            )
        else:
            print("Vector store already exists with data. Skipping initialization.")
            vectorstore = Qdrant(
                client=client, 
                collection_name=COLLECTION_NAME, 
                embeddings=embeddings
            )
    except Exception as e:
        print(f"Error checking collection: {e}")
        print("Attempting to create vector store...")
        vectorstore = initialize_vectorstore(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
            texts=text_chunks
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
        print(f"Processing query: {query}")
        # Use the correct input format for RetrievalQA
        response = qa_chain.invoke({"query": query})
        print(f"Response keys: {response.keys()}")
        
        # Try different possible keys for the result
        if "result" in response:
            return response["result"]
        elif "answer" in response:
            return response["answer"]
        else:
            return str(response)
            
    except Exception as e:
        print(f"Error running query: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error processing your query. Please try again."
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from srcode.utilites import load_repo, document_loader_repo, split_documents

load_dotenv()

# Setup API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "source-code-analysis-rag-v1"

# 1. Check and Create Index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


# Clone a repository
repo_url = "https://github.com/Ahmed2797/PneumoScan-AI.git"
repo_path = load_repo(url=repo_url)

# Load documents from the cloned repo
docs = document_loader_repo(repo_path)
print(f"Loaded {len(docs)} documents")
chunks = split_documents(documents=docs, chunk_size=100, chunk_overlap=20)

# 3. Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. Load documents into Pinecone via LangChain
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print("Ingestion complete. Documents are now searchable in Pinecone.")
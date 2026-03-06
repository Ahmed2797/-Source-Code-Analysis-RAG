from srcode.retrive import GenerateRetriever
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from srcode.utilites import load_repo, document_loader_repo, split_documents
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = OpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

index_name = "source-code-analysis-rag-v1"


# Clone a repository
repo_url = "https://github.com/Ahmed2797/PneumoScan-AI.git"
repo_path = load_repo(url=repo_url)

# Load documents from the cloned repo
docs = document_loader_repo(repo_path)
print(f"Loaded {len(docs)} documents")
chunks = split_documents(documents=docs, chunk_size=100, chunk_overlap=20)

# 3. Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

gen_retriever = GenerateRetriever(vector_store=vector_store)
memory = gen_retriever.create_memory(llm=llm)
chain = gen_retriever.create_conversational_chain(llm=llm, memory=memory)

question = "What is the main functionality of the code in this repository?"
response = chain.invoke(question)
print(f"Question: {question}")
print(f"Answer: {response['answer']}")


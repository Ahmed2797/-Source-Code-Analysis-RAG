import streamlit as st
import os
from dotenv import load_dotenv

# Import your custom modules
from srcode.retrive import GenerateRetriever
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from srcode.utilites import load_repo, document_loader_repo, split_documents

# --- Page Configuration ---
st.set_page_config(page_title="PneumoScan Code Analyzer", page_icon="💻", layout="wide")
load_dotenv()

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatFloatingInputContainer { bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- App Logic & State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_backend():
    """Initializes the vector store and retrieval chain once."""
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    index_name = "source-code-analysis-rag-v1"
    
    # Initialize components
    llm = OpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    gen_retriever = GenerateRetriever(vector_store=vector_store)
    memory = gen_retriever.create_memory(llm=llm)
    chain = gen_retriever.create_conversational_chain(llm=llm, memory=memory)
    
    return chain

# --- Sidebar: Repository Management ---
with st.sidebar:
    st.title("🛠 Settings")
    repo_url = st.text_input("Repository URL")
    
    if st.button("Re-index Repository"):
        with st.status("Indexing repository...", expanded=True) as status:
            st.write("Cloning...")
            repo_path = load_repo(url=repo_url)
            st.write("Loading Docs...")
            docs = document_loader_repo(repo_path)
            st.write("Splitting Chunks...")
            chunks = split_documents(documents=docs, chunk_size=100, chunk_overlap=20)
            # Note: You might want to add vector_store.add_documents(chunks) here 
            # if the index isn't already populated.
            status.update(label="Index Ready!", state="complete", expanded=False)
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat UI ---
st.title("📂 Source Code Analysis RAG")
st.caption("Chat with your codebase to understand logic, structure, and functionality.")

# Initialize the chain
try:
    chain = initialize_backend()
except Exception as e:
    st.error(f"Failed to connect to Pinecone/OpenAI: {e}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me anything about the code..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Analyzing code..."):
            response = chain.invoke(prompt)
            answer = response['answer']
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

## streamlit run app.py
## https://github.com/Ahmed2797/PneumoScan-AI.git
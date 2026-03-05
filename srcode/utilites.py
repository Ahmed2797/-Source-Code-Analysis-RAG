import os
from git import Repo
from typing import List
from langchain_classic.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_classic.text_splitter import Language
# from langchain_classic.document_loaders.generic import GenericLoader
# from langchain_classic.document_loaders.parsers import LanguageParser

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser


def load_repo(url: str = None, repo_dir: str = "github_repo") -> str:
    """
    Clone a git repository to a local directory.
    
    Args:
        url (str): Git repository URL
        repo_dir (str): Local directory to clone into
        
    Returns:
        str: Path to the cloned repository
    """
    if not url:
        raise ValueError("Repository URL is required")
    
    # Create directory if it doesn't exist
    os.makedirs(repo_dir, exist_ok=True)
    
    # Clone the repository
    repo_name = url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(repo_dir, repo_name)
    
    # Check if repo already exists
    if not os.path.exists(repo_path):
        print(f"Cloning repository from {url} to {repo_path}")
        Repo.clone_from(url=url, to_path=repo_path)
    else:
        print(f"Repository already exists at {repo_path}")
    
    return repo_path


def document_loader_repo(path: str):
    """
    Load Python documents from a repository path.
    
    Args:
        path (str): Path to the repository
        
    Returns:
        list: Loaded document objects
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    # Create loader for Python files
    loader = GenericLoader.from_filesystem(
        path=path,
        glob="**/*",
        suffixes=['.py'],
        parser=LanguageParser(
            language=Language.PYTHON,
            parser_threshold=200  # Increased from 100 for better chunking
        )
    )
    
    # Load documents
    documents = loader.load()
    
    return documents

# # Example usage:
# if __name__ == "__main__":
#     # Clone a repository
#     repo_url = "https://github.com/Ahmed2797/Network-Security.git"
#     repo_path = load_repo(url=repo_url)
    
#     # Load documents from the cloned repo
#     docs = document_loader_repo(repo_path)
#     print(f"Loaded {len(docs)} documents")



def split_documents(documents: List[Document], chunk_size: int = 100, chunk_overlap: int = 20) -> List[Document]:
    """
    Split a list of LangChain Document objects into smaller chunks.

    Args:
        documents (List[Document]): List of loaded documents (PDF pages, etc.)
        chunk_size (int, optional): Size of each text chunk. Defaults to 100.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 20.

    Returns:
        List[Document]: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Chunks created: {len(chunks)}")
    return chunks



def embed_text(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """
    Generate embeddings for a given text using a HuggingFace Sentence-Transformer model.

    Args:
        model_name (str, optional): The HuggingFace model to use. Defaults (384) to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        List[float]: The embedding vector as a list of floats.

    """
    embeddings = HuggingFaceEmbeddings(model=model_name)

    return embeddings


def load_openai_embeddings(model_name: str = "text-embedding-3-small"):
    """
    Generate embeddings for a given text using a OpenAIEmbeddings model.

    Args:
        model_name (str, optional): The OpenAIEmbeddings model to use. Defaults to 'text-embedding-3-small'.

    Returns:
        List[float]: The embedding vector as a list of floats.

    """
    return OpenAIEmbeddings(
        model=model_name
        # model="text-embedding-3-small"
    )

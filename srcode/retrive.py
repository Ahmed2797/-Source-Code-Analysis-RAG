from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.vectorstores import VectorStore
from langchain_classic.schema import BaseRetriever


class GenerateRetriever:
    def __init__(self, vector_store: VectorStore, memory_key: str = 'chat_history'):
        """
        Initialize a conversational retriever with memory.
        
        Args:
            vector_store: Vector store containing embedded documents
            memory_key: Key for storing chat history in memory
        """
        self.vector_store = vector_store
        self.memory_key = memory_key  
    
    def create_memory(self, llm) -> ConversationSummaryMemory:
        """
        Create conversation summary memory.
        
        Args:
            llm: Language model instance
            
        Returns:
            ConversationSummaryMemory: Configured memory object
        """
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key=self.memory_key,
            return_messages=True,output_key="answer"
        )
        return memory
    
    def create_conversational_chain(self, llm, memory: ConversationSummaryMemory) -> ConversationalRetrievalChain:
        """
        Create conversational retrieval chain.
        
        Args:
            llm: Language model instance
            memory: Conversation memory object
            
        Returns:
            ConversationalRetrievalChain: Configured retrieval chain
        """
        # Create retriever from vector store
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,           # Number of documents to return
                "fetch_k": 50,    # Number of documents to fetch for MMR
                "score_threshold": 0.5  # Optional: minimum similarity score
            }
        )
        
        # Create conversational retrieval chain
        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            #verbose=True,  # Shows what's happening
            chain_type="stuff",  # Options: "stuff", "map_reduce", "refine", "map_rerank"
            return_source_documents=True  # Return source documents for citation
            
        )
        
        print(f"Created conversational chain with retriever")
        print(f"Search type: MMR, k: 5, fetch_k: 50")
        
        return conversational_chain
    
    def simple_retriever(self, search_type: str = "similarity", k: int = 4) -> BaseRetriever:
        """
        Create a simple retriever without conversation chain.
        
        Args:
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            k: Number of documents to retrieve
            
        Returns:
            BaseRetriever: Configured retriever
        """
        search_kwargs = {"k": k}
        
        # Add additional parameters based on search type
        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = 0.7
        
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return retriever


# if __name__=='__main__':
#     gen_retriver = GenerateRetriever(load_vector_store)
#     memory = gen_retriver.create_memory(llm=llm)
#     chain = gen_retriver.create_conversational_chain(llm=llm,memory=memory)
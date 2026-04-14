import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from loader import load_documents, chunk_documents

# Step 1: Create embeddings model (free, runs on your laptop)
def get_embeddings():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Small, fast, free model
    )
    print("Embedding model ready.")
    return embeddings

# Step 2: Build and save the vector store
def build_vectorstore(chunks, embeddings):
    print(f"\nConverting {len(chunks)} chunks into vectors...")
    print("This may take 1-2 minutes on first run...")
    
    # Set the folder where ChromaDB will save the vectors
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(base_dir, "vectorstore_db")
    
    # Create the vector store and save it to disk
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"Vector store built and saved to: {persist_dir}")
    return vectorstore

# Step 3: Load existing vector store (for future use)
def load_vectorstore(embeddings):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(base_dir, "vectorstore_db")
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectorstore

# Step 4: Test it
if __name__ == "__main__":
    # Load and chunk documents
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_dir, "documents")
    
    documents = load_documents(docs_path)
    chunks = chunk_documents(documents)
    
    # Build vector store
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    
    # Test similarity search
    print("\n--- Testing Similarity Search ---")
    query = "What is the GDP of Pakistan?"
    results = vectorstore.similarity_search(query, k=3)
    
    print(f"Query: {query}")
    print(f"\nTop 3 most relevant chunks found:\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(doc.page_content[:200])  # Show first 200 chars
        print()
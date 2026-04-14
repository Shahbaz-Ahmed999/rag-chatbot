import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Step 1: Load all PDF files from the documents folder
def load_documents(folder_path):
    all_documents = []
    
    # Loop through every file in the documents folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):  # Only process PDF files
            file_path = os.path.join(folder_path, filename)
            print(f"Loading: {filename}")
            
            loader = PyPDFLoader(file_path)  # Read the PDF
            documents = loader.load()        # Convert to text
            all_documents.extend(documents)  # Add to our list
    
    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents


# Step 2: Split documents into small chunks
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Each chunk = 500 characters
        chunk_overlap=50,     # 50 characters overlap between chunks
        length_function=len
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


# Step 3: Test it — run this file directly to see output
if __name__ == "__main__":
    # Go up one folder from src/ to reach documents/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_dir, "documents")
    
    # Load and chunk
    documents = load_documents(docs_path)
    chunks = chunk_documents(documents)
    
    # Print first 2 chunks so you can see what they look like
    print("\n--- Sample Chunk 1 ---")
    print(chunks[0].page_content)
    print("\n--- Sample Chunk 2 ---")
    print(chunks[1].page_content)
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vectorstore import get_embeddings, load_vectorstore

# Load the API key from .env file
load_dotenv()

# Step 1: Connect to Groq LLaMA-3
def get_llm():
    print("Connecting to Groq LLaMA-3...")
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=512
    )
    print("LLM ready.")
    return llm

# Step 2: Format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 3: Build the full RAG chain
def build_rag_chain():
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = get_llm()

    prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer.
If the answer is not clearly stated in the context, say "I don't have enough information in my documents to answer this clearly." Never repeat phrases or sentences.
Give a concise and direct answer. Do not repeat any words, phrases, or sentences.

Context:
{context}

Question:
{question}

Answer:
""")

    # Modern LangChain chain using LCEL (pipe syntax)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain ready.\n")
    return chain, retriever

# Step 4: Test with 5 real questions
if __name__ == "__main__":
    chain, retriever = build_rag_chain()

    test_questions = [
        "What is the GDP of Pakistan?",
        "What does the State Bank of Pakistan do?",
        "What is the role of SECP in Pakistan?",
        "What are the main exports of Pakistan?",
        "What is the inflation rate in Pakistan?"
    ]

    for question in test_questions:
        print(f"Question: {question}")
        answer = chain.invoke(question)
        print(f"Answer: {answer}")

        # Show source documents
        docs = retriever.invoke(question)
        sources = set(os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs)
        print(f"Sources: {', '.join(sources)}")
        print("-" * 60)
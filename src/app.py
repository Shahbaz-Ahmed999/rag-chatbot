import os
import sys
import streamlit as st

# Make sure src/ folder is accessible
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag_chain import build_rag_chain

# Page configuration
st.set_page_config(
    page_title="Pakistani Finance Q&A Assistant",
    page_icon="🏦",
    layout="centered"
)

# Title and description
st.title("🏦 Pakistani Finance Q&A Assistant")
st.markdown("""
Ask any question about **Pakistan's Economy, State Bank of Pakistan, or SECP**.
This assistant answers using real documents — not guesswork.
""")

st.divider()

# Load the RAG chain once and cache it
# This prevents reloading on every question
@st.cache_resource
def load_chain():
    with st.spinner("Loading AI model... (first load takes ~30 seconds)"):
        chain, retriever = build_rag_chain()
    return chain, retriever

chain, retriever = load_chain()

# Question input
question = st.text_input(
    "💬 Ask a question:",
    placeholder="e.g. What is the GDP of Pakistan?"
)

# Submit button
if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        # Show spinner while processing
        with st.spinner("Searching documents and generating answer..."):
            answer = chain.invoke(question)
            docs = retriever.invoke(question)
            sources = set(
                os.path.basename(doc.metadata.get("source", "Unknown"))
                for doc in docs
            )

        # Display answer
        st.subheader("📋 Answer")
        st.write(answer)

        # Display sources
        st.subheader("📄 Sources")
        for source in sources:
            st.info(f"📁 {source}")
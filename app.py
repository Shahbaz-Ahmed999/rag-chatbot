import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag_chain import build_rag_chain

st.set_page_config(
    page_title="Pakistan Finance Assistant",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
<style>
.answer-box {
    background-color: var(--background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-left: 4px solid #1a7f4b;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-top: 0.5rem;
    line-height: 1.7;
}
.source-pill {
    display: inline-block;
    background-color: rgba(26,127,75,0.1);
    color: #1a7f4b;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.8rem;
    margin-right: 6px;
    margin-top: 4px;
}
.chunk-box {
    background-color: rgba(128,128,128,0.05);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🏦 Pakistan Finance Assistant")
    st.markdown("""
**RAG-powered Q&A** from real Pakistani financial documents.

---
**Documents loaded:**
- Economy of Pakistan
- State Bank of Pakistan  
- SECP Pakistan

---
**Tech stack:**
- LangChain + ChromaDB
- LLaMA-3 via Groq
- HuggingFace Embeddings
- Streamlit

---
""")
    st.caption("💡 Tip: ask specific questions for best results.")
    st.caption("Built as a portfolio project · [GitHub](#)")

st.title("🏦 Pakistan Finance Q&A")
st.caption("Ask about Pakistan's economy, banking, or capital markets — answers sourced directly from documents.")
st.divider()

@st.cache_resource(show_spinner="Loading AI model, please wait...")
def load_chain(force_rebuild=False):
    return build_rag_chain(force_rebuild=force_rebuild)

if st.sidebar.button("🔄 Rebuild Knowledge Base"):
    st.cache_resource.clear()
    st.rerun()

chain, retriever = load_chain()

st.markdown("**Try a sample question:**")
col1, col2, col3, col4 = st.columns(4)
if col1.button("GDP?", use_container_width=True):
    st.session_state.q = "What is the GDP of Pakistan?"
if col2.button("SBP role?", use_container_width=True):
    st.session_state.q = "What does the State Bank of Pakistan do?"
if col3.button("SECP role?", use_container_width=True):
    st.session_state.q = "What is the role of SECP in Pakistan?"
if col4.button("Exports?", use_container_width=True):
    st.session_state.q = "What are the main exports of Pakistan?"

if "q" not in st.session_state:
    st.session_state.q = ""

question = st.text_input(
    "Your question:",
    value=st.session_state.q,
    placeholder="e.g. What is the inflation rate in Pakistan?"
)

if st.button("Get Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            answer = chain.invoke(question)
            docs = retriever.invoke(question)
            sources = set(
                os.path.basename(doc.metadata.get("source", "Unknown"))
                for doc in docs
            )

        st.markdown("#### Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        st.markdown("#### Sources")
        source_html = "".join(f'<span class="source-pill">📄 {s}</span>' for s in sources)
        st.markdown(source_html, unsafe_allow_html=True)

        st.markdown("")
        with st.expander("View retrieved document chunks"):
            for i, doc in enumerate(docs):
                name = os.path.basename(doc.metadata.get("source", "Unknown"))
                st.markdown(f"**Chunk {i+1}** — `{name}`")
                st.markdown(f'<div class="chunk-box">{doc.page_content[:300]}...</div>',
                            unsafe_allow_html=True)
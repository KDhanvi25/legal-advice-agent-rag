import os
import pickle
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ====== Paths ======
INDEX_PATH = "legal_faiss.index"
CHUNKS_PATH = "legal_chunks.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/phi-2"

# ====== Load FAISS index and chunks ======
@st.cache_resource
def load_index_and_chunks():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks, sources = pickle.load(f)
    return index, chunks, sources

# ====== Load Embedding and LLM ======
@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto", torch_dtype="auto")
    qa_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return embedder, qa_pipe

# ====== Retrieve Relevant Chunks ======
def get_top_chunks(query, k=5):
    q_vec = embedder.encode([query])
    D, I = index.search(q_vec, k)
    return [(chunks[i], sources[i]) for i in I[0]]

# ====== Prompt Construction ======
def build_prompt(query, history):
    retrieved = get_top_chunks(query)
    
    # 1. Cleaner context without <chunk> tags
    context = "\n".join([f"{chunk}" for chunk, _ in retrieved])

    # 2. Chat history (no formatting change here)
    history_block = "\n".join([f"User: {h[0]}\nAgent: {h[1]}" for h in history])

    # 3. Bold user question using Markdown-style formatting
    prompt = f"""You are a highly reliable legal assistant that answers based on Indian law only.
Use all relevant details from the sources below. Be exhaustive, cite legal principles, case examples, and explain clearly like a good lawyer would.

{context}

{history_block}
User: **{query}**
Agent:"""

    return prompt, retrieved

# ====== Streamlit UI ======
st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️")
st.title("⚖️ Indian Law QA Agent")

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.rerun()

query = st.text_input("Ask a legal question", key="user_input")

if query:
    index, chunks, sources = load_index_and_chunks()
    embedder, qa_pipe = load_models()

    with st.spinner("Thinking..."):
        # Only use last 5 exchanges from chat history
        trimmed_history = st.session_state.history[-5:]
        prompt, retrieved = build_prompt(query, trimmed_history)
        result = qa_pipe(prompt, max_new_tokens=1024, do_sample=True)[0]["generated_text"]
        answer = result[len(prompt):].strip()

        # Store in memory
        st.session_state.history.append((query, answer))

        # Show answer
        st.markdown("### 🔍 Legal Answer")
        st.markdown(answer)

        # Show sources
        st.markdown("### 📚 Sources")
        unique_sources = sorted(set(src for _, src in retrieved))
        for src in unique_sources:
            st.markdown(f"- `{src}`")

# Show full chat history (all turns)
if st.session_state.history:
    with st.expander("💬 Chat History"):
        for q, a in st.session_state.history[::-1]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

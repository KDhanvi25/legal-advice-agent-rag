import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# === Load everything ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("legal_faiss.index")

with open("legal_chunks.pkl", "rb") as f:
    chunks, sources = pickle.load(f)

# === Choose LLM ===
# For local Phi-2:
# model_name = "microsoft/phi-2"
# For remote GPT-3.5 (OpenAI):
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
qa_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

# === RAG Function ===
def get_relevant_chunks(query, k=5):
    q_vec = model.encode([query])
    D, I = index.search(q_vec, k)
    return [(chunks[i], sources[i]) for i in I[0]]

def ask_legal_agent(query):
    retrieved = get_relevant_chunks(query)
    # Add inline source tags to context
    context_blocks = [f"[Source: {src}]\n{text}" for text, src in retrieved]
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful legal assistant trained on Indian law (IPC, CrPC, Constitution, and FAQs).
Answer the user's question using only the context below. Cite the source when relevant.

{context}

Question: {query}
Answer:"""

    output = qa_pipeline(prompt, max_new_tokens=512, do_sample=True)[0]["generated_text"]

    print("\n🔍 Top Legal Answer:\n")
    print(output[len(prompt):].strip())

    # Also print summary of sources used
    sources_used = list(set(src for _, src in retrieved))
    print("\n📚 Sources used:")
    for src in sources_used:
        print(f" - {src}")

# === CLI Interface ===
if __name__ == "__main__":
    while True:
        query = input("\n⚖️  Ask your legal question (or type 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        ask_legal_agent(query)

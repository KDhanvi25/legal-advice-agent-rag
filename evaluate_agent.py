import csv
import re
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np

# ====== Setup ======
INDEX_PATH = "legal_faiss.index"
CHUNKS_PATH = "legal_chunks.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/phi-2"

# Load FAISS & chunks
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks, sources = pickle.load(f)

embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto", torch_dtype="auto")
qa_pipe = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

# Benchmark questions
evaluation_questions = [ 
    "What does Section 420 of the IPC say?",
    "What are the grounds for divorce under Hindu Marriage Act?",
    "Can a woman claim alimony after divorce in India?",
    "What is the punishment for defamation in Indian law?",
    "What is anticipatory bail and when can it be granted?",
    "Is attempt to suicide a criminal offence in India?",
    "What is Article 21 of the Constitution of India?",
    "Can a minor be tried for murder in India?",
    "What is the difference between bailable and non-bailable offences?",
    "What are the rights of a daughter in ancestral property?",
    "What is the procedure to file an FIR?",
    "Is live-in relationship legally recognized in India?",
    "What is the maternity leave entitlement under Indian law?",
    "Can an OCI cardholder buy property in India?",
    "What is the POCSO Act and whom does it protect?",
    "Can WhatsApp chats be used as evidence in Indian courts?",
    "What is the maximum time police can detain a person without producing them in court?",
    "What is Section 498A of IPC about?",
    "Can a Hindu man marry a Christian woman in India?",
    "What is the difference between Article 14 and Article 15 of the Constitution?"
] 

# Legal keywords for coverage metric
legal_terms = ["IPC", "CrPC", "Section", "Article", "Act", "Constitution", "Court"]

# ====== Helper Functions ======
def get_top_chunks(query, k=5):
    q_vec = embedder.encode([query])
    D, I = index.search(q_vec, k)
    return [chunks[i] for i in I[0]]

def build_prompt(query):
    retrieved = get_top_chunks(query)
    context = "\n".join(retrieved)
    prompt = f"""Answer the following question based only on Indian law.

{context}

User: {query}
Agent:"""
    return prompt

def evaluate_response(query, response):
    # Length
    length = len(response.split())

    # Semantic similarity (cosine between question and answer embeddings)
    q_emb = embedder.encode([query])
    a_emb = embedder.encode([response])
    cos_sim = float(np.dot(q_emb, a_emb.T) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb)))

    # Legal term coverage
    terms_found = sum(1 for term in legal_terms if re.search(rf"\b{term}\b", response, re.IGNORECASE))

    return length, cos_sim, terms_found

# ====== Main Eval ======
with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Query", "Response", "Length", "CosineSim", "LegalTermsCount"])

    for q in evaluation_questions:
        prompt = build_prompt(q)
        result = qa_pipe(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        answer = result[len(prompt):].strip()

        length, cos_sim, terms_found = evaluate_response(q, answer)

        writer.writerow([q, answer, length, cos_sim, terms_found])
        print(f"✅ {q[:40]}... | Len: {length} | CosSim: {cos_sim:.2f} | Terms: {terms_found}")

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

TEXT_FOLDER = "formatted_txts"
INDEX_PATH = "legal_faiss.index"
CHUNKS_PATH = "legal_chunks.pkl"

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Map from file names to real source names
source_map = {
    "ipc_formatted.txt": "Indian Penal Code (IPC)",
    "crpc_formatted.txt": "Criminal Procedure Code (CrPC)",
    "coi_formatted.txt": "Constitution of India",
    "nyaaya_faqs.txt": "https://nyaaya.org/question-bank/"
}

# Load chunks with proper source tags
def load_chunks(folder):
    chunks = []
    sources = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                blocks = f.read().strip().split("\n\n")
                readable_source = source_map.get(file, file)  # fallback to filename
                for block in blocks:
                    chunks.append(block.strip())
                    sources.append(readable_source)
    return chunks, sources

# Main process
chunks, sources = load_chunks(TEXT_FOLDER)
print(f"✅ Loaded {len(chunks)} text chunks from {len(set(sources))} sources.")

# Embed
print("📌 Embedding with MiniLM...")
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and sources
faiss.write_index(index, INDEX_PATH)
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump((chunks, sources), f)

print(f"✅ FAISS index saved to {INDEX_PATH}")
print(f"✅ Chunk metadata saved to {CHUNKS_PATH}")

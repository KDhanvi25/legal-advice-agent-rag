⚖️ RAG-based Legal Q&A System for Indian Law

A Retrieval-Augmented Generation (RAG) system designed to answer Indian law questions with relevant context and clear legal reasoning.
The system retrieves sections from the IPC, CrPC, Constitution of India, and Nyaaya FAQs, then uses Microsoft Phi‑2 (2.7B) to generate lawyer‑style answers.

📂 Project Structure
.
├── app.py                 # Streamlit UI for interactive querying
├── build_index.py         # Builds FAISS index from formatted legal text files
├── evaluate_agent.py      # Automated evaluation on benchmark questions
├── query_agent.py         # CLI interface for asking legal questions
├── requirements.txt       # Python dependencies (see note for Streamlit)
├── legal_faiss.index      # FAISS index of embedded chunks
├── legal_chunks.pkl       # Pickled chunks and their sources
├── evaluation_results.csv # Saved evaluation metrics
├── results.txt            # Console output from an evaluation run
├── pdfs/                  # Raw legal documents (PDFs)
└── formatted_txts/        # Cleaned & formatted text files for indexing


🚀 Features
PDF → text ingestion (curated statutes + FAQs)
Embeddings with all-MiniLM-L6-v2
Dense retrieval via FAISS
LLM answers with Microsoft Phi‑2
Two interfaces: Streamlit app + CLI
Evaluation: response length, cosine similarity, and legal-term coverage


🛠️ Installation
git clone https://github.com/yourusername/legal-advice-agent.git
cd legal-advice-agent
pip install -r requirements.txt
# For the web app UI:
pip install streamlit


📊 Build the FAISS Index
Put cleaned text files in formatted_txts/ and run:
python build_index.py

This creates:
legal_faiss.index
legal_chunks.pkl

💬 Run the Agent
streamlit run app.py
Open http://localhost:8501, then ask:

“What is anticipatory bail and when can it be granted?”

“What are the rights of a daughter in ancestral property?”

📈 Evaluation
python evaluate_agent.py

Outputs:
evaluation_results.csv (per‑query metrics)
Console summary (example in results.txt)

📊 Results (Exact Averages from 20 Benchmark Questions)
Average Response Length: 345.1 words
Average Cosine Similarity (Q ↔ A): 0.689
Average Legal-Term Coverage: 1.4 terms/answer
(terms counted from: IPC, CrPC, Section, Article, Act, Constitution, Court)
These metrics show consistent, context‑aware answers with legal terminology appearing regularly in responses.

📦 Requirements
From requirements.txt:
torch
sentence-transformers
faiss-gpu
transformers
accelerate

If you use the web app:
streamlit

📚 Sources Used
Indian Penal Code (IPC)
Criminal Procedure Code (CrPC)
Constitution of India
Nyaaya FAQs (https://nyaaya.org/question-bank/)

⚠️ Disclaimer
This project is for informational purposes only and does not constitute legal advice. Please consult a qualified lawyer for actual legal matters.
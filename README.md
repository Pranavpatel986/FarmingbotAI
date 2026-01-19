---
title: "FarmerBot AI"
emoji: "🌾"
colorFrom: "green"
colorTo: "lime"
sdk: "gradio"
sdk_version: "6.3.0"
app_file: "app.py"
pinned: false
description: "FarmerBot AI: Advanced Conversational RAG for Scientific Agriculture"
---

# 🌾 FarmerBot AI: Advanced Conversational RAG for Scientific Agriculture

FarmerBot AI is a state-of-the-art Retrieval-Augmented Generation (RAG) system designed to act as a Senior Scientific Agricultural Advisor. Developed as part of a **Master of Technology in AI & Data Science at MNNIT Allahabad**, this project leverages a multi-stage hybrid retrieval pipeline to provide expert-level, grounded advice based on the *Farmer's Handbook on Basic Agriculture*.



## 📐 Detailed Project Architecture

The system is built on a modular **Advanced RAG** pipeline that prioritizes factual accuracy and scientific grounding over generic AI responses.

### 1. Data Engineering & Ingestion
* **Source:** Scientific PDF documentation (*Farmer's Handbook*).
* **Chunking Strategy:** Implemented `RecursiveCharacterTextSplitter` with a **chunk size of 800 characters** and a **100-character overlap** to preserve semantic continuity.
* **Vectorization:** Utilizes `all-MiniLM-L6-v2` (local HuggingFace model) to generate 384-dimensional dense embeddings.
* **Persistence:** A pre-indexed **ChromaDB** instance is shipped with the repository to ensure zero-latency cold starts on cloud environments.

### 2. The Hybrid Retrieval Engine
To solve the limitation of vector-only search (which often misses specific technical terms), FarmerBot employs an **Ensemble Retrieval** strategy:
* **Dense Search (Vector):** Uses cosine similarity to find conceptually related information.
* **Sparse Search (BM25):** A keyword-based algorithm that ensures 100% accuracy for specific terms (e.g., "Azotobacter", "NPK", "Urea").
* **Weighted Fusion:** Combines results using a $0.6$ Vector and $0.4$ Keyword weighting.

### 3. Neural Reranking & Refinement
* **Flashrank Integration:** After initial retrieval, the system uses a **Neural Reranker**.
* **Re-scoring:** Documents are re-evaluated against the query to ensure the most relevant context is prioritized, significantly reducing LLM hallucinations.

### 4. Generative Reasoning
* **Model:** **Google Gemini 2.5 Flash** (2026 Stable Standard).
* **Context Window:** Strategically limits the context to the top 3-5 high-relevance chunks.
* **Conversational Memory:** Implements a dictionary-based history formatter compatible with **Gradio 6.3.0**.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language Model** | Google Gemini 2.5 Flash |
| **Orchestration** | LangChain (Modular 1.x Architecture) |
| **Vector Database** | ChromaDB (Persistent) |
| **Embeddings** | HuggingFace `sentence-transformers` |
| **Reranker** | Flashrank (Neural Cross-Encoder) |
| **Keyword Search** | Rank_BM25 |
| **UI Framework** | Gradio (Glassmorphic Custom Theme) |
| **Deployment** | Hugging Face Spaces (16GB RAM Optimized) |

---

## ⚙️ Installation & Usage

### 1. Setup Environment
```bash
git clone [https://github.com/yourusername/farmingbot.git](https://github.com/yourusername/farmingbot.git)
cd farmingbot
python -m venv venv
# Activate (Windows): .\venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Set API Key

Create a `.env` file locally (Note: On Hugging Face, set this in **Settings > Secrets**):

```env
GOOGLE_API_KEY=your_gemini_api_key_here

```

### 3. Run Application

```bash
python app.py

```

---

## 🌐 Deployment Logic

This project is deployed on **Hugging Face Spaces** to leverage superior hardware for local AI models.

* **Hardware:** Running on **CPU Basic (2 vCPU, 16GB RAM)**.
* **Storage:** Uses **Git LFS** to manage the large `chroma_db` index and the `farmerbook.pdf`.
* **Deployment Workflow:** Uses a **Dual-Remote Git Strategy** (`origin` for GitHub, `space` for Hugging Face).
* **CI/CD:** Automated builds triggered via `git push space main`.

---

## 🔮 Future Roadmap

* **Multimodal Diagnosis:** Integrating **Gemini 2.0 Vision** for crop disease photo analysis.
* **Multilingual Support:** Translation layers for Hindi, Bengali, and other regional languages.
* **Agentic Workflows:** Weather and Market Price agents using live API tools.

---

## 👨‍🔬 Author

**Pranav Patel** *M.Tech in Artificial Intelligence & Data Science* *Motilal Nehru National Institute of Technology (MNNIT), Allahabad*

```

---


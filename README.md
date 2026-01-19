---

# 🌾 FarmerBot AI: Advanced Conversational RAG for Scientific Agriculture

**FarmerBot AI** is a state-of-the-art **Retrieval-Augmented Generation (RAG)** system designed to act as a Senior Scientific Agricultural Advisor. Developed as part of a **Master of Technology in AI & Data Science at MNNIT Allahabad**, this project leverages a multi-stage hybrid retrieval pipeline to provide expert-level, grounded advice based on the *Farmer's Handbook on Basic Agriculture*.

## 📐 Detailed Project Architecture

The system is built on a modular **Advanced RAG** pipeline that prioritizes factual accuracy and scientific grounding over generic AI responses.

### 1. Data Engineering & Ingestion

* **Source:** Scientific PDF documentation (*Farmer's Handbook*).
* **Chunking Strategy:** Implemented `RecursiveCharacterTextSplitter` with a **chunk size of 800 characters** and a **100-character overlap** to preserve semantic continuity between fragments.
* **Vectorization:** Utilizes `all-MiniLM-L6-v2` (local HuggingFace model) to generate 384-dimensional dense embeddings.
* **Persistence:** A pre-indexed **ChromaDB** instance is shipped with the repository to ensure zero-latency cold starts on cloud environments.

### 2. The Hybrid Retrieval Engine (Modern Implementation)

To solve the limitation of vector-only search (which often misses specific technical terms or chemical names), FarmerBot employs an **Ensemble Retrieval** strategy:

* **Semantic Search (Dense):** Uses cosine similarity to find conceptually related information.
* **BM25 Search (Sparse):** An industry-standard keyword-based algorithm that ensures 100% accuracy for specific names (e.g., "Urea", "NPK", "Rhizobium").
* **Weighted Fusion:** Combines results using a  Vector and  Keyword weighting.

### 3. Neural Reranking & Refinement

* **Flashrank Integration:** After the initial retrieval of the top 10 documents, the system uses a **Neural Reranker**.
* **Re-scoring:** Documents are re-evaluated against the query to ensure the most relevant context is prioritized in the prompt's first few tokens, significantly reducing LLM hallucinations.

### 4. Generative Reasoning

* **Model:** **Google Gemini 2.5 Flash** (2026 Stable Standard).
* **Context Window Management:** Strategically limits the context to the top 3-5 high-relevance chunks to maintain speed and stay within API TPM limits.
* **Conversational Memory:** Implements a dictionary-based history formatter compatible with **Gradio 6.3.0** message structures.

---

## 🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| **Language Model** | Google Gemini 2.5 Flash |
| **Orchestration** | LangChain (Modular 1.x Architecture) |
| **Vector Database** | ChromaDB (Persistent) |
| **Embeddings** | HuggingFace `sentence-transformers` |
| **Reranker** | Flashrank (Neural Cross-Encoder) |
| **Keyword Search** | Rank_BM25 |
| **UI Framework** | Gradio (Glassmorphic Custom Theme) |
| **Deployment** | Render Cloud (via Blueprint & Build.sh) |

---

## ⚙️ Installation & Usage

### 1. Setup Environment

```bash
git clone https://github.com/yourusername/farmingbot.git
cd farmingbot
python -m venv venv
# Activate (Windows): .\venv\Scripts\activate | (Linux): source venv/bin/activate
pip install -r requirements.txt

```

### 2. Set API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here

```

### 3. Run Application

```bash
python chatbot.py

```

---

## 🌐 Deployment Logic

The project is optimized for **Render's Free Tier** using an Infrastructure-as-Code (IaC) approach.

* **Build Cache Optimization:** The `build.sh` script detects the pre-built `chroma_db` folder to skip redundant ingestion cycles, saving build minutes and RAM.
* **Dynamic Port Binding:** Uses `os.environ.get("PORT")` to allow Render's load balancer to map traffic to the Gradio server.
* **Environment Sync:** Securely handles the Google API Key via Render's `sync: false` blueprint setting.

---

## 🔮 Future Roadmap

* **Multimodal Diagnosis:** Integrating **Gemini 2.0 Vision** to allow farmers to upload photos of diseased crops for visual diagnosis.
* **Multilingual Support:** Implementing real-time translation layers for regional Indian languages (Hindi, Bengali, etc.).
* **Agentic Workflows:** Adding a "Weather Agent" tool to provide irrigation advice based on live local weather APIs.

---

## 👨‍🔬 Author

**Pranav Patel**
*M.Tech in Artificial Intelligence & Data Science*
*Motilal Nehru National Institute of Technology (MNNIT), Allahabad*

---
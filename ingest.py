import os
import shutil
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()

def build_db():
    print("🚀 Starting PDF Ingestion with LOCAL Embeddings...")
    
    # Path settings
    # Note: Using DirectoryLoader will find all PDFs, but we check for farmerbook.pdf specifically first
    pdf_path = os.path.join("data", "farmerbook.pdf")
    db_path = "chroma_db"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found! Please ensure your PDF is in the 'data' folder.")
        return

    # 1. Clean old database if it exists
    if os.path.exists(db_path):
        print("🧹 Cleaning old database...")
        shutil.rmtree(db_path)

    # 2. Load documents
    # Fixed Indentation here
    print("📄 Loading PDFs...")
    loader = DirectoryLoader('data/', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    # 3. Split Text
    # Fixed variable name from 'pages' to 'docs' to match the loader output
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks.")

    # 4. Initialize Local Embeddings
    print("🧠 Initializing local embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 5. Build and Persist Vector Store
    print("📡 Building Vector Store... (This may take a minute depending on your CPU)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="farmer_kb"
    )

    print(f"✅ SUCCESS: Database built successfully in '{db_path}'!")

if __name__ == "__main__":
    build_db()
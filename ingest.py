import os
import shutil
import stat
from dotenv import load_dotenv

# Core LangChain 2026 Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()

def remove_readonly(func, path, excinfo):
    """Handles Windows PermissionError by clearing the readonly bit."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def build_db():
    print("🚀 Starting Batch PDF Ingestion...")
    
    data_dir = "data"
    db_path = "chroma_db"
    
    # 1. Check if the data folder exists and has PDFs
    if not os.path.exists(data_dir):
        print(f"❌ Error: Folder '{data_dir}' not found!")
        return
    
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDF files found in '{data_dir}'.")
        return
    
    print(f"📂 Found {len(pdf_files)} PDFs: {', '.join(pdf_files)}")

    # 2. Clean old database (Standard procedure for fresh re-indexing)
    if os.path.exists(db_path):
        print("🧹 Cleaning old database...")
        try:
            shutil.rmtree(db_path, onerror=remove_readonly)
        except Exception as e:
            print(f"⚠️ Manual action required: Delete the '{db_path}' folder. Reason: {e}")
            return

    # 3. Load ALL documents from the directory
    print("📄 Loading all documents...")
    # The DirectoryLoader will now iterate through every .pdf in the folder
    loader = DirectoryLoader(
        data_dir, 
        glob="./*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    try:
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages from {len(pdf_files)} files.")
    except Exception as e:
        print(f"❌ Error during loading: {e}")
        return
    
    # 4. Split Text into Semantic Chunks
    # Using a slightly smaller overlap for better diversity across multiple documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=80,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} chunks.")

    # 5. Initialize Embeddings (Sentence Transformers)
    print("🧠 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 6. Build Vector Store
    print("📡 Indexing chunks into ChromaDB... Please wait.")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="farmer_kb"
    )

    print(f"🏁 SUCCESS: All {len(pdf_files)} PDFs are indexed in '{db_path}'!")

if __name__ == "__main__":
    build_db()
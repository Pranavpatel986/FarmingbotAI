import os
from dotenv import load_dotenv
import gradio as gr

# Core AI & Local Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# --- CLASSIC IMPORTS FOR STABILITY ---
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.retrievers import BM25Retriever

# UI Configuration
from ui_config import custom_css, hero_html

load_dotenv()

# --- 1. BACKEND INITIALIZATION ---
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "farmer_kb"


# To this (The 2026 Stable Standard):
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
    max_retries=3,
    timeout=60
)

# Global variable to cache the retriever
advanced_retriever = None

def get_advanced_retriever():
    # Keep your exact logic, just moved inside the function
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    all_data = vector_store.get()
    
    if not all_data['documents']:
        return vector_retriever 
        
    docs = [Document(page_content=d, metadata=m) for d, m in zip(all_data['documents'], all_data['metadatas'])]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.4, 0.6]
    )

    compressor = FlashrankRerank()
    return ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble
    )

# --- 3. CONVERSATIONAL SYSTEM PROMPT ---
qa_system_prompt = """
You are "FarmerBot", a Senior Scientific Agricultural Advisor.
Use the provided context and chat history to provide accurate, data-driven advice.

### GUIDELINES:
1. **Source Grounding:** Answer ONLY using the provided Context. 
2. **Scientific Persona:** Use bold text for key terms and bullet points for steps.
3. **Safety:** Advise consulting a local Agricultural Officer for chemical emergencies.

### CONTEXT:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# --- 4. CORE CHAT LOGIC ---
def format_history(history):
    formatted_history = []
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            if role == "user":
                formatted_history.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_history.append(AIMessage(content=content))
        elif isinstance(entry, (list, tuple)):
            user, bot = entry
            formatted_history.append(HumanMessage(content=user))
            formatted_history.append(AIMessage(content=bot))
    return formatted_history

def farmer_chat(message, history):
    global advanced_retriever
    query = message["text"] if isinstance(message, dict) else message
    
    try:
        # LAZY LOADING: Initialize retriever only on the first message
        if advanced_retriever is None:
            yield "⏳ FarmerBot is warming up and reading the Handbook... Please wait."
            advanced_retriever = get_advanced_retriever()

        chat_history = format_history(history)
        docs = advanced_retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        chain = qa_prompt | llm | StrOutputParser()
        
        response = ""
        for chunk in chain.stream({
            "chat_history": chat_history,
            "context": context_text, 
            "question": query
        }):
            response += chunk
            yield response
            
    except Exception as e:
        yield f"❌ System Error: {str(e)}"

# --- 5. GRADIO INTERFACE ---
with gr.Blocks(title="FarmerBot AI") as demo: 
    gr.HTML(hero_html)
    
    with gr.Column(elem_id="floating_container"):
        gr.HTML('<div class="widget-header"><span>🌱</span> FarmerBot Advisor</div>')
        
        gr.ChatInterface(
            fn=farmer_chat,
            examples=["How to test soil?", "Management of Rice pests"],
            cache_examples=False
        )

# --- 6. DEPLOYMENT SETTINGS ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 FarmerBot binding to port {port}...")
    
    # demo.queue() is crucial for Render to maintain the connection
    demo.queue().launch(
        server_name="127.0.0.1", 
        server_port=port,
        share=False,
        css=custom_css
    )
import os
from dotenv import load_dotenv
import gradio as gr

# Light imports for fast initial load
load_dotenv()
from ui_config import custom_css, hero_html

# Global variable to cache the retriever
advanced_retriever = None

def get_advanced_retriever():
    # HEAVY IMPORTS MOVED INSIDE (Lazy Load)
    from langchain_huggingface import HuggingFaceEmbeddings 
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    # Use standard langchain path for these stable retrievers
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    from langchain_community.retrievers import BM25Retriever

    CHROMA_PATH = "chroma_db"
    COLLECTION_NAME = "farmer_kb"
    
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
Use the provided context to provide accurate, data-driven advice.

### GUIDELINES:
1. **Source Grounding:** Answer ONLY using the provided Context. 
2. **Scientific Persona:** Use bold text for key terms and bullet points for steps.
3. **Safety:** Advise consulting a local Agricultural Officer for chemical emergencies.

### CONTEXT:
{context}
"""

# --- 4. CORE CHAT LOGIC ---
def format_history(history):
    from langchain_core.messages import HumanMessage, AIMessage
    formatted_history = []
    for entry in history:
        # Gradio 6.x often uses dict format
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            # Ensure content is string (handles Gradio 6 content blocks)
            if isinstance(content, list):
                content = " ".join([c["text"] for c in content if "text" in c])
            
            if role == "user":
                formatted_history.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_history.append(AIMessage(content=content))
        # Fallback for list/tuple format
        elif isinstance(entry, (list, tuple)):
            user, bot = entry
            formatted_history.append(HumanMessage(content=user))
            formatted_history.append(AIMessage(content=bot))
    return formatted_history

def farmer_chat(message, history):
    global advanced_retriever
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser

    query = message["text"] if isinstance(message, dict) else message
    
    try:
        if advanced_retriever is None:
            yield "⏳ FarmerBot is warming up and reading the Handbook... Please wait."
            advanced_retriever = get_advanced_retriever()

        # Recommended stable model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

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

# --- 6. HUGGING FACE LAUNCH ---
if __name__ == "__main__":
    # Standard launch for Hugging Face Spaces (automatically uses port 7860)
    demo.queue().launch()
import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# RAGAS Imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain & App Imports
from app import get_advanced_retriever, qa_system_prompt
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. INITIALIZATION
load_dotenv()

print("🌱 Initializing FarmerBot RAG Components...")
# This uses your exact Ensemble + Flashrank logic from app.py
advanced_retriever = get_advanced_retriever()

# Initialize Gemini for the RAG pipeline
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Gemini Wrappers for Ragas (The "Judge")
# This prevents the OpenAI API Key error
ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(embeddings)

# Define the evaluation chain
prompt = ChatPromptTemplate.from_template("""
{system_prompt}
CONTEXT: {context}
QUESTION: {question}
""")
rag_chain = prompt | llm | StrOutputParser()

# 2. EVALUATION DATASET (Ground Truth from farmerbook.pdf)
data = {
    "question": [
        "What is the primary focus of the Farmer's Handbook on Basic Agriculture?",
        "What are the components of Integrated Nutrient Management (INM)?",
        "How should farmers handle pesticides according to the safety guidelines?",
        "What is the importance of soil testing mentioned in the handbook?",
        "What does the handbook say about farm management and its objectives?"
    ],
    "ground_truth": [
        "The handbook provides a holistic perspective of scientific agriculture to impart technical knowledge on basic agriculture to farmers.",
        "INM involves the use of soil testing, plant nutrition requirements, and a combination of organic and inorganic fertilizers.",
        "Farmers should follow safe handling practices, use protective gear, understand toxicity categories, and practice proper storage and disposal.",
        "Soil testing is essential for understanding soil health, fertility, and determining the correct amount of nutrients needed for sustainable management.",
        "Farm management focuses on economically sustainable management and increasing awareness of farm resources for better productivity."
    ]
}

# 3. GENERATION PHASE
def run_evaluation_pipeline(questions):
    answers = []
    contexts = []
    
    for q in questions:
        print(f"🧐 Processing Question: {q}")
        # Retrieve context using your Ensemble + Rerank logic
        docs = advanced_retriever.invoke(q)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # Generate Answer
        response = rag_chain.invoke({
            "system_prompt": qa_system_prompt,
            "context": context_text,
            "question": q
        })
        
        answers.append(response)
        # Ragas expects a list of strings for each context
        contexts.append([d.page_content for d in docs])
        
    return answers, contexts

print("🏃 Running RAG Pipeline to collect responses...")
answers, contexts = run_evaluation_pipeline(data["question"])

# 4. RAGAS METRIC CALCULATION
dataset = Dataset.from_dict({
    "question": data["question"],
    "answer": answers,
    "contexts": contexts,
    "ground_truth": data["ground_truth"]
})

print("📊 Calculating RAGAS Metrics using Gemini as the Judge...")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_emb
)

# 5. FINAL OUTPUT
df = result.to_pandas()
print("\n--- FINAL EVALUATION SCORES ---")
print(df)

df.to_csv("farmerbot_evaluation_results.csv", index=False)
print("\n✅ Results saved to farmerbot_evaluation_results.csv")
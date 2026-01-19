#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# ONLY run ingestion if the folder is missing 
# (This acts as a safety net if you ever delete the folder from GitHub)
if [ ! -d "chroma_db" ]; then
    echo "📂 ChromaDB folder not found in repo. Running ingestion..."
    python ingest.py
else
    echo "✅ Using pre-built ChromaDB found in repository."
fi
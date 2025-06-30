import json
import os
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/bge-small-en" # Or all-MiniLM-L6-v2
CHUNK_SIZE = 256 # Smaller chunks for free tier efficiency
CHUNK_OVERLAP = 30 # Minimal overlap to save resources

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def load_text_from_file(filepath):
    """Loads text from a given file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def parse_cve_data(filepath):
    """
    Parses a simplified CVE JSON.
    Expected format: [{"id": "CVE-XXXX-YYYY", "description": "...", "severity": "..."}]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        cves = json.load(f)
    docs = []
    for cve in cves:
        content = f"CVE ID: {cve.get('id', 'N/A')}\nSeverity: {cve.get('severity', 'N/A')}\nDescription: {cve.get('description', '')}"
        metadata = {"source": "CVE", "id": cve.get('id')}
        docs.append({"content": content, "metadata": metadata})
    return docs

def parse_mitre_attack_data(filepath):
    """
    Parses a simplified MITRE ATT&CK JSON.
    Expected format: [{"id": "TXXXX", "name": "...", "description": "..."}]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        techniques = json.load(f)
    docs = []
    for tech in techniques:
        content = f"ATT&CK ID: {tech.get('id', 'N/A')}\nName: {tech.get('name', 'N/A')}\nDescription: {tech.get('description', '')}"
        metadata = {"source": "MITRE ATT&CK", "id": tech.get('id')}
        docs.append({"content": content, "metadata": metadata})
    return docs

def parse_advisory_html(filepath):
    """Parses text from HTML advisories."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            # Extract main text content, e.g., from paragraphs or specific divs
            main_content = soup.get_text(separator='\n', strip=True)
            # You might need to refine this based on actual advisory HTML structure
            return main_content
    except Exception as e:
        print(f"Error parsing HTML {filepath}: {e}")
        return ""

def parse_advisory_txt(filepath):
    """Parses text from plain text advisories."""
    return load_text_from_file(filepath)


def load_cybersecurity_data():
    """Loads and preprocesses all cybersecurity data."""
    all_docs = []

    # 1. Load CVEs
    cve_filepath = os.path.join(DATA_DIR, "cves.json")
    if os.path.exists(cve_filepath):
        print(f"Loading CVE data from {cve_filepath}...")
        all_docs.extend(parse_cve_data(cve_filepath))
    else:
        print(f"Warning: {cve_filepath} not found. Skipping CVE data.")

    # 2. Load MITRE ATT&CK
    mitre_filepath = os.path.join(DATA_DIR, "mitre_attack.json")
    if os.path.exists(mitre_filepath):
        print(f"Loading MITRE ATT&CK data from {mitre_filepath}...")
        all_docs.extend(parse_mitre_attack_data(mitre_filepath))
    else:
        print(f"Warning: {mitre_filepath} not found. Skipping MITRE ATT&CK data.")

    # 3. Load Advisories/Blogs
    advisories_dir = os.path.join(DATA_DIR, "advisories")
    if os.path.exists(advisories_dir):
        print(f"Loading advisories from {advisories_dir}...")
        for filename in os.listdir(advisories_dir):
            filepath = os.path.join(advisories_dir, filename)
            if filename.endswith(".html"):
                content = parse_advisory_html(filepath)
            elif filename.endswith(".txt"):
                content = parse_advisory_txt(filepath)
            else:
                continue # Skip other file types

            if content:
                all_docs.append({"content": content, "metadata": {"source": "Advisory", "filename": filename}})
    else:
        print(f"Warning: {advisories_dir} not found. Skipping advisory data.")
    
    # 4. Load Glossary (as simple document chunks)
    glossary_filepath = os.path.join(DATA_DIR, "glossary.txt")
    if os.path.exists(glossary_filepath):
        print(f"Loading glossary from {glossary_filepath}...")
        glossary_content = load_text_from_file(glossary_filepath)
        all_docs.append({"content": glossary_content, "metadata": {"source": "Glossary"}})
    else:
        print(f"Warning: {glossary_filepath} not found. Skipping glossary data.")

    return all_docs

def create_vector_store(documents):
    """Chunks documents, embeds them, and saves to FAISS."""
    if not documents:
        print("No documents to process. Exiting.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunks = []
    metadata = []
    for doc in tqdm(documents, desc="Chunking documents"):
        splits = text_splitter.create_documents([doc["content"]], metadatas=[doc["metadata"]])
        for split in splits:
            chunks.append(split.page_content)
            metadata.append(split.metadata)

    if not chunks:
        print("No chunks created after splitting. Exiting.")
        return

    print(f"Total chunks created: {len(chunks)}")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Generating embeddings (this may take a while for large datasets)...")
    # Batch processing for embeddings to manage memory
    embeddings = embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Save embedding model for later verification/reloading if needed (optional)
    # embedding_model.save_pretrained(os.path.join(VECTOR_STORE_DIR, "embeddings", EMBEDDING_MODEL_NAME.replace("/", "_")))

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss_index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
    faiss_metadata_path = os.path.join(VECTOR_STORE_DIR, "faiss_metadata.json")

    print(f"Saving FAISS index to {faiss_index_path}")
    faiss.write_index(index, faiss_index_path)

    print(f"Saving metadata to {faiss_metadata_path}")
    with open(faiss_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("Vector store creation complete.")

if __name__ == "__main__":
    print("Starting data preprocessing for RAG...")
    all_raw_docs = load_cybersecurity_data()
    if all_raw_docs:
        create_vector_store(all_raw_docs)
    else:
        print("No data found or loaded. Please ensure your 'data/' directory contains the necessary files.")
    print("Data preprocessing finished.")

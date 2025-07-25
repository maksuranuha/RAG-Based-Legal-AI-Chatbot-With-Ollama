from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
import shutil

data_directory = 'data/'

def upload_pdf(file_path):
    os.makedirs(data_directory, exist_ok=True)
    filename = os.path.basename(file_path)
    destination = os.path.join(data_directory, filename)
    shutil.copy2(file_path, destination)
    return destination

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    return docs

def load_multiple_pdfs(file_paths):
    all_docs = []
    for path in file_paths:
        if os.path.exists(path):
            try:
                docs = load_pdf(path)
                all_docs.extend(docs)
                print(f"Loaded: {os.path.basename(path)} - {len(docs)} pages")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"Missing: {path}")
    return all_docs

pdf_files = [
    'data/Digital-Security-Act-2018.pdf',
]

print("Loading PDFs...")
documents = load_multiple_pdfs(pdf_files)
print(f"Total documents loaded: {len(documents)}")

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    return chunks

print("Creating chunks...")
text_chunks = create_chunks(documents)
print(f"Chunks count: {len(text_chunks)}")

# tetsing ollama
try:
    embeddings = OllamaEmbeddings(model="mistral:latest")
    test_embedding = embeddings.embed_query("test")
    print(f"Ollama connection successful - embedding dimension: {len(test_embedding)}")
except Exception as e:
    print(f"Ollama connection failed: {e}")
    print("Make sure Ollama is running and mistral model is available")
    exit(1)

DB_PATH = "vectorstore/db_faiss"
os.makedirs("vectorstore", exist_ok=True)

print("Building knowledge base...")
print("This may take several minutes for 103 chunks...")

try:
    #avoiding memory issues
    batch_size = 20
    all_vectors = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1} ({len(batch)} chunks)")
        
        if i == 0:
            vector_db = FAISS.from_documents(batch, embeddings)
        else:
            batch_db = FAISS.from_documents(batch, embeddings)
            vector_db.merge_from(batch_db)
    
    vector_db.save_local(DB_PATH)
    print(f"Knowledge base successfully saved to {DB_PATH}")
    
except Exception as e:
    print(f"Error building knowledge base: {e}")
    print("Try reducing chunk_size or batch_size if memory issues persist")
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import shutil
import os
from dotenv import load_dotenv
load_dotenv() 

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set in environment")

llm = ChatGroq(model="mistral-saba-24b", api_key=api_key)

# Step 1: Question Processing
def process_question(user_question):
    return user_question.strip()

# FIXED: Consistent function definition
def get_embeddings(model_name="mistral:latest"):
    return OllamaEmbeddings(model=model_name)

# Step 3: Knowledge Source Management
data_dir = 'data/'

def handle_file_upload(file_path): 
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.basename(file_path)
    destination = os.path.join(data_dir, filename)
    shutil.copy2(file_path, destination)
    return destination

def load_document(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])
    return [Document(page_content=full_text)]

# Step 4: Text Chunking (Chunk 1...Chunk N)
def split_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=False
    )
    return text_splitter.split_documents(documents)

# Step 5: Create Embeddings (Embedding 1...EmbeddingN)
def get_embedding_model(model_name="mistral:latest"):
    return get_embeddings(model_name)

# Step 6: Knowledge Base Creation
DB_PATH = "vectorstore/db_faiss"

def build_knowledge_base(chunks, model_name="mistral:latest"):
    embeddings = get_embedding_model(model_name)
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_PATH)
    return vector_db

def load_knowledge_base(model_name="mistral:latest"):
    if os.path.exists(DB_PATH):
        embeddings = get_embeddings(model_name)
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# Step 7: Semantic Search & Ranking
def search_knowledge_base(question, vector_db=None):
    if vector_db is None:
        vector_db = load_knowledge_base()
        if vector_db is None:
            return []
    
    relevant_docs = vector_db.similarity_search(question, k=5)
    return relevant_docs

legal_prompt = """
If user says formal greetings like Hi, Hello, bye then Reply user respectfully with similar greetings! 
Use the provided legal context to answer the user's question about Bangladesh law.
If you don't know the answer from the context, say so clearly.

Don't make up legal advice outside the given context.

Question: {question}
Legal Context: {context}

Answer:
"""

summarizer_chain = load_summarize_chain(llm, chain_type="map_reduce")

def get_context_from_docs(docs, max_context_words=2000):
    if not docs:
        return "No relevant legal context found."

    total_words = sum(len(doc.page_content.split()) for doc in docs)

    if total_words > max_context_words:
        print("Condensing context with summarization...")
        try:
            summary = summarizer_chain.run(docs)
            if hasattr(summary, 'content'):
                return summary.content.strip()
            return str(summary).strip()
        except Exception as e:
            print(f"Summarization failed: {e}")
            return "\n\n".join([doc.page_content[:1000] for doc in docs])  # fallback short context
    else:
        return "\n\n".join([doc.page_content for doc in docs])

def generate_answer(question, relevant_docs):
    context = get_context_from_docs(relevant_docs)
    prompt = ChatPromptTemplate.from_template(legal_prompt)
    chain = prompt | llm
    
    response = chain.invoke({
        "question": question, 
        "context": context
    })
    
    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, str):
        return response
    else:
        response_str = str(response)
        if 'content=' in response_str:
            import re
            content_match = re.search(r"content='([^']*(?:''[^']*)*)'", response_str)
            if content_match:
                return content_match.group(1).replace("''", "'")
        return response_str

def is_answer_sufficient(answer):
    weak_indicators = [
        "Sorry, don't know", "I dont know", "no information", 
        "not mentioned", "cannot find", "not available in the context",
        "context does not contain", "i don't have", "not provided"
    ]
    answer_text = str(answer).lower()
    return not any(indicator in answer_text for indicator in weak_indicators)

# FALLBACK SEARCHES 
# Option 1: Bangladesh Laws Website
def search_bd_laws(query):
    try:
        url = "http://bdlaws.minlaw.gov.bd/laws-of-bangladesh-alphabetical-index.html"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.get_text()
        
        relevant_lines = []
        for line in content.split('\n'):
            if any(word.lower() in line.lower() for word in query.split()):
                relevant_lines.append(line.strip())
        
        return '\n'.join(relevant_lines[:5]) if relevant_lines else ""
    except:
        return ""

# Option 2: Tavily Search 
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    tavily_client = None

def search_tavily(query):
    if not tavily_client:
        return "Add valid tavily API key to .env file"
    
    try:
        response = tavily_client.search(query + " Bangladesh law legal")
        return response.get('answer', '')
    except Exception as e:
        return f"Tavily search failed: {str(e)}"

def search_online_fallback(question):
    bd_result = search_bd_laws(question)
    tavily_result = search_tavily(question)
    
    fallback_info = ""
    if bd_result:
        fallback_info += f"From Bangladesh Laws: {bd_result}\n\n"
    if tavily_result and "not configured" not in tavily_result:
        fallback_info += f"External Search: {tavily_result}"
    
    return fallback_info.strip()

def run_rag_pipeline(user_question, uploaded_file=None):
    
    # Step 1: Process question from chatbot
    question = process_question(user_question)
    
    # Step 2-3: Handle file upload if provided
    if uploaded_file:
        # uploaded_file is the file path from Gradio
        file_destination = handle_file_upload(uploaded_file)
        docs = load_document(file_destination)
        chunks = split_into_chunks(docs)
        vector_db = build_knowledge_base(chunks)
    else:
        vector_db = load_knowledge_base()
    
    # Step 4-7: Semantic search and ranking
    relevant_docs = search_knowledge_base(question, vector_db)
    
    if not relevant_docs:
        return "No relevant documents found in knowledge base."
    
    # Step 8: LLM Answer
    llm_answer = generate_answer(question, relevant_docs)
    
    if is_answer_sufficient(llm_answer):
        return llm_answer
    else:
        print("LLM answer insufficient, trying online search...")
        online_info = search_online_fallback(question)
        
        if online_info:
            return f"{llm_answer}\n\n Additional Information from Online Sources \n{online_info}"
        else:
            return f"{llm_answer}\n\n(No additional information found from online sources)"
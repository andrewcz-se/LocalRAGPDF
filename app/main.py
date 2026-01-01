import os
os.environ["LANGCHAIN_DISABLE_TELEMETRY"] = "true"
import sys
import json
import time
import requests
import hashlib
import pdfplumber
from typing import Dict, List, Any, Tuple

from flask import Flask, render_template, request, jsonify

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

app = Flask(__name__)

PDF_FOLDER = "pdfs"
CHROMA_DIR = "chroma_data"
STATE_FILE = "ingested_state.json"

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# Global DB reference
vector_db = None

# ---------------------------------------------------------
# Hashing & State Management
# ---------------------------------------------------------

def calculate_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except FileNotFoundError:
        return ""

def load_state() -> Dict[str, str]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_state(state: Dict[str, str]):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# ---------------------------------------------------------
# Ollama Setup
# ---------------------------------------------------------

def wait_for_ollama(timeout=60):
    print("Waiting for Ollama to be ready...")
    start = time.time()
    while True:
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
            if r.status_code == 200:
                print("✓ Ollama is ready.")
                return True
        except:
            pass
        if time.time() - start > timeout:
            print("ERROR: Ollama did not become ready in time.")
            return False
        time.sleep(1)

def ensure_ollama_model(model_name: str):
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags")
        models = r.json().get("models", [])
        if any(m.get("name") == model_name for m in models):
            print(f"✓ Model '{model_name}' already installed.")
            return
    except:
        pass

    print(f"↓ Model '{model_name}' not found. Downloading...")
    try:
        requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model_name})
        print(f"✓ Model '{model_name}' installed.")
    except Exception as e:
        print(f"Error downloading model: {e}")

# ---------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------

def extract_sections_from_pdf(path) -> Tuple[List[tuple], Dict[int, List[tuple]]]:
    """
    Returns:
    1. sections: List of (title, text, page)
    2. spatial_map: Dict { page_num: [ (y_coord, section_path), ... ] }
       Note: y_coord is from PDFMiner (Bottom-Up: 0 is bottom, Height is top)
    """
    sections = []
    current_text = []
    current_page = 1
    
    heading_stack = [] 
    default_title = "Introduction"
    
    # Store list of headers per page to resolve multiple sections on one page
    spatial_map = {} 
    
    SIZE_TOLERANCE = 0.3

    try:
        for page_layout in extract_pages(path):
            page_height = page_layout.height
            top_margin = page_height * 0.90
            bottom_margin = page_height * 0.10
            
            # Initialize list for this page
            if current_page not in spatial_map:
                spatial_map[current_page] = []

            # 1. Capture "Start of Page" Context
            # If the page starts with body text (before any new header), 
            # it belongs to the active section from the previous page.
            # We treat this as a header at y = Infinity (top of page).
            if heading_stack:
                active_section = " > ".join([t for s, t in heading_stack])
            else:
                active_section = default_title
            
            # Add a "virtual header" at the very top of the page (y = page_height + 1)
            spatial_map[current_page].append((page_height + 1, active_section))

            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    if element.y0 > top_margin: continue
                    if element.y1 < bottom_margin: continue

                    text = element.get_text().strip()
                    if not text: continue
                    
                    font_sizes = []
                    bold_count = 0
                    for line in element:
                        for char in line:
                            if isinstance(char, LTChar):
                                size = char.size or 0
                                font_sizes.append(size)
                                if "bold" in char.fontname.lower(): bold_count += 1
                    
                    avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 10
                    total_chars = len(font_sizes)
                    bold_ratio = bold_count / total_chars if total_chars else 0
                    
                    is_heading = avg_font > 12 or (avg_font > 10 and bold_ratio > 0.4)

                    if is_heading:
                        if heading_stack:
                            full_path = " > ".join([t for s, t in heading_stack])
                        else:
                            full_path = default_title
                        
                        if current_text:
                            full_section = " ".join(current_text)
                            sections.append((full_path, full_section, current_page))
                            current_text = []

                        while heading_stack and heading_stack[-1][0] < (avg_font + SIZE_TOLERANCE):
                            heading_stack.pop()
                        
                        heading_stack.append((avg_font, text))
                        
                        # RECORD HEADER POSITION
                        # element.y0 is the bottom of the text line. 
                        # Any table below this (smaller y) belongs to this header.
                        new_path = " > ".join([t for s, t in heading_stack])
                        spatial_map[current_page].append((element.y0, new_path))

                    else:
                        clean_line = text.replace('\n', ' ')
                        current_text.append(clean_line)
            current_page += 1
            
        if current_text:
            if heading_stack:
                full_path = " > ".join([t for s, t in heading_stack])
            else:
                full_path = default_title
            full_section = " ".join(current_text)
            sections.append((full_path, full_section, current_page - 1))
            
    except Exception as e:
        print(f"Warning parsing {path}: {e}")
        
    return sections, spatial_map

def extract_full_pdf_text(path):
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

# ---------------------------------------------------------
# Table Extraction
# ---------------------------------------------------------

def table_to_markdown(table: List[List[str]]) -> str:
    if not table or not table[0]: return ""
    cleaned_table = []
    for row in table:
        cleaned_row = [str(cell).replace('\n', ' ') if cell is not None else "" for cell in row]
        cleaned_table.append(cleaned_row)
    headers = cleaned_table[0]
    separator = ["---"] * len(headers)
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(separator) + " |")
    for row in cleaned_table[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)

def extract_tables_with_positions(path) -> List[tuple]:
    """
    Returns list of (markdown, page_num, table_top_y_coord)
    """
    extracted_tables = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # use find_tables to get the bounding box
                tables = page.find_tables()
                if tables:
                    for table_obj in tables:
                        # table_obj.bbox = (x0, top, x1, bottom) 
                        # pdfplumber uses "top-down" y (0 is top of page)
                        # We need to convert to pdfminer "bottom-up" y (0 is bottom of page)
                        
                        pdfplumber_top = table_obj.bbox[1] 
                        page_height = page.height
                        
                        # Convert to bottom-up Y coordinate to match pdfminer
                        table_y_coord = page_height - pdfplumber_top
                        
                        # Extract data
                        table_data = table_obj.extract()
                        md_table = table_to_markdown(table_data)
                        
                        if len(md_table) > 20: 
                            extracted_tables.append((md_table, i + 1, table_y_coord))
    except Exception as e:
        print(f"Warning extracting tables from {path}: {e}")
    return extracted_tables

# ---------------------------------------------------------
# Vector DB Logic
# ---------------------------------------------------------

def get_vector_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def resolve_section_for_table(page_num, table_y, spatial_map):
    """
    Finds the closest header that is ABOVE the table.
    """
    if page_num not in spatial_map:
        return "Unknown Section"
    
    headers = spatial_map[page_num]
    # headers list is [(y_coord, title), ...]
    # We want the header with the smallest Y that is still > table_y 
    # (Since higher Y means higher up on the page)
    
    candidate_section = "Unknown Section"
    
    # Sort headers by Y position descending (Top to Bottom)
    # Example: [ (800, 'Title'), (600, 'Subtitle'), (400, 'Footer') ]
    sorted_headers = sorted(headers, key=lambda x: x[0], reverse=True)
    
    for (header_y, title) in sorted_headers:
        if header_y > table_y:
            candidate_section = title
        else:
            # If we hit a header below the table, we stop.
            # The previous one was the correct parent.
            break
            
    return candidate_section

def sync_documents(db):
    print("Syncing documents...")
    state = load_state()
    new_state = {}
    
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        
    folder_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
        #separators=[""\n***\n", "***", "\n\n", "\n", ". ", " ", ""]
    )
    
    for filename in folder_files:
        filepath = os.path.join(PDF_FOLDER, filename)
        current_hash = calculate_file_hash(filepath)
        
        if filename in state and state[filename] == current_hash:
            new_state[filename] = current_hash
            continue
            
        print(f"→ Processing {filename}")
        if filename in state:
             try:
                 ids = db.get(where={"file": filename})['ids']
                 if ids: db.delete(ids=ids)
             except: pass

        # 1. Text Extraction (Returns Spatial Map)
        sections, spatial_map = extract_sections_from_pdf(filepath)
        full_text = extract_full_pdf_text(filepath)
        
        documents = []
        
        for (title, text, page) in sections:
            raw_chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(raw_chunks):
                enhanced_content = f"# Source: {filename}\n## Section: {title}\n\n{chunk}"
                doc = Document(
                    page_content=enhanced_content,
                    metadata={
                        "file": filename, "section": title, "page": page, "type": "text", "chunk_index": i
                    }
                )
                documents.append(doc)
        
        # 2. Table Extraction (With Coordinates)
        print(f"   - Scanning for tables...")
        tables = extract_tables_with_positions(filepath)
        
        if tables:
            print(f"   - Found {len(tables)} tables.")
            for i, (md_table, page, table_y) in enumerate(tables):
                
                # SPATIAL RESOLUTION
                active_section = resolve_section_for_table(page, table_y, spatial_map)
                
                enhanced_content = f"# Source: {filename}\n## Section: {active_section}\n### Table Data\n\n{md_table}"
                
                doc = Document(
                    page_content=enhanced_content,
                    metadata={
                        "file": filename, "section": active_section, "page": page, "type": "table", "chunk_index": i
                    }
                )
                documents.append(doc)
            
        if documents:
            db.add_documents(documents)
            print(f"   - Added {len(documents)} total chunks.")
        
        new_state[filename] = current_hash

    save_state(new_state)
    print("✓ Sync complete.\n")

def perform_rag(query: str) -> Dict[str, Any]:
    global vector_db
    if not vector_db:
        return {"error": "Database not initialized"}

    #retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 50, "lambda_mult": 0.8}
    )

    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_HOST)

    template = """You are a detailed and informative assistant.

Context:
{context}

Question: 
{question}

Instructions:
1. Answer the question comprehensively based on the Context. Include all relevant details found in the documents.
2. Do NOT explain your search process. Do NOT say "The context specifies..." or "No other entities found...". Just present the facts directly.
3. If the context mentions multiple entities (e.g. animal vs vehicle) with the same name, provide full details for both clearly.
4. Analyze the ENTIRE context before coming to a final answer.
5. If the answer is NOT in the context, output exactly: "I do not know based on the provided documents."

Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa.invoke({"query": query})
    
    # Extract Context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc in result["source_documents"]])
    
    # Extract Sources list
    seen = set()
    sources_list = []
    for doc in result["source_documents"]:
        meta = doc.metadata
        identifier = f"{meta.get('file')}-{meta.get('section')}"
        if identifier not in seen:
            sources_list.append({
                "file": meta.get("file"),
                "section": meta.get("section"),
                "page": meta.get("page")
            })
            seen.add(identifier)

    return {
        "answer": result["result"],
        "context": context_text,
        "sources": sources_list
    }

# ---------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    # Gather stats
    state = load_state()
    files = list(state.keys())
    
    # Count total chunks (native chroma call)
    try:
        chunk_count = vector_db._collection.count()
    except:
        chunk_count = 0
        
    return render_template("index.html", files=files, chunk_count=chunk_count)

@app.route("/query", methods=["POST"])
def query_route():
    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
        
    response = perform_rag(user_query)
    return jsonify(response)

# --- NEW: Inspector Route ---
@app.route("/inspect", methods=["GET"])
def inspect_route():
    if not vector_db:
        return jsonify({"error": "Database not initialized"}), 500
    
    # Optional: Filter by filename
    filename = request.args.get('file')
    limit = int(request.args.get('limit', 20))
    
    if filename and filename != "all":
        results = vector_db.get(where={"file": filename}, limit=limit)
    else:
        results = vector_db.get(limit=limit)
    
    # Reformat for frontend
    data = []
    if results and results['documents']:
        for i, doc in enumerate(results['documents']):
            data.append({
                "id": results['ids'][i],
                "content": doc,
                "metadata": results['metadatas'][i]
            })
            
    return jsonify(data)

# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

def init_app():
    global vector_db
    if wait_for_ollama():
        ensure_ollama_model(EMBED_MODEL)
        ensure_ollama_model(LLM_MODEL)
        vector_db = get_vector_db()
        sync_documents(vector_db)

if __name__ == "__main__":
    init_app()
    # Host 0.0.0.0 is required for Docker
    app.run(host="0.0.0.0", port=5000, debug=False)
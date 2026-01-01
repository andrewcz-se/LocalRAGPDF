# **Local PDF RAG Assistant**

A local RAG (Retrieval-Augmented Generation) application. The application is a self hosted **Retrieval-Augmented Generation (RAG)** system designed to chat with your PDF documents. The tool allows you to index PDF files into a local vector database and query them using local LLMs via Ollama, with a specific focus on maintaining the relationship between text, tables, and section headers. It combines a clean Flask web interface with a backend that parses, chunks, and retrieves information from local files using Ollama and ChromaDB.

The project features a basic **context-aware ingestion pipeline** designed to handle complex documents like manuals, textbooks, and reports where context is easily lost across chunk boundaries.

## **Key Features**

* **Spatial PDF Parsing:** The application maps the physical layout of the PDF. It links tables to the specific section headers they appear under by comparing vertical coordinates.  
* **Tabular Data Support:** Automatically detects tables using pdfplumber and converts them into Markdown format, ensuring the LLM understands the structural relationships within the data.  
* **Intelligent State Management:** Uses MD5 hashing to track file changes. Only new or modified PDFs are processed, significantly reducing ingestion time for large libraries.  
* **Advanced Retrieval (MMR):** Implements **Maximal Marginal Relevance (MMR)** to fetch diverse context chunks, reducing redundancy in the information provided to the LLM.  
* **Database Inspector:** A built-in web interface to browse the raw chunks, metadata (page numbers, sections), and IDs stored in your ChromaDB.  
* **Context-Aware Chunking:** Intelligently injects file and section headers into *every* individual text chunk, ensuring the LLM never loses track of what it is reading (even in the middle of long lists).  
* **Layout Analysis:** Uses pdfminer to visually inspect font sizes and boldness to detect document structure (headers vs. body text) and filters out noise like page numbers and running footers.  
* **Strict Factual Prompting:** System prompt designed to prevent the model from "waffling" or explaining its search process, while encouraging detailed answers for ambiguous terms (e.g., "Puma" the car vs. "Puma" the animal).  

## **How It Works**

### **Ingestion**

This application uses a multi-stage process:

1. **State Check:** On startup, the app checks pdfs/ against a local state file (ingested_state.json). It calculates MD5 hashes and identifies which files are new, modified, or deleted.  
2. **Visual Parsing:** Instead of just extracting raw text, I used some basic **Layout Analysis**:  
   * **Noise Filtering:** Ignore the top 10% and bottom 10% of every page to strip out headers, footers, and page numbers that confuse vector search.  
   * **Header Detection:** Scan for text that is either larger than 12pt or significantly bolded. These are treated as "Section Titles."  
3. **Context Injection (The "Per-Chunk" Strategy):**  
   * Split the text first, then **prepend the File Name and Section Title to every single chunk.**  
   * *Result:* A chunk containing bullet point #50 still explicitly says: # Source: manual.pdf ## Section: Safety Protocols. This allows the LLM to answer questions about specific details without losing context.

### **RAG**

When you ask a question via the Web UI:

1. **Vector Search:** The query is embedded using nomic-embed-text and sent to ChromaDB. The **Top 15** most relevant chunks are retrieved (increased from initial standard 4-5 to handle "broad" questions like "Tell me about planets").  
2. **Prompt Assembly:** The retrieved chunks are stitched together into a Context block.  
3. **LLM Inference:** The qwen:4b model (running locally via Ollama) processes the prompt.  
   * **Persona:** "Detailed and informative."  
   * **Constraints:** The model is explicitly forbidden from using "meta-talk" (e.g., "The context mentions..."). It must answer directly.  
   * **Conflict Resolution:** If the context contains multiple entities with the same name (e.g., Puma the animal vs. Puma the car model), the prompt instructs the model to describe *both*.

## **Tech Stack**

* **Frontend:** HTML5, Bootstrap 5, JavaScript (Fetch API).  
* **Server:** Python Flask.  
* **Orchestration:** Docker Compose.  
* **LLM Serving:** Ollama (running qwen:4b).  
* **Embeddings:** nomic-embed-text.  
* **Vector Database:** ChromaDB (persistent local storage).  
* **PDF Processing:** pdfminer.six (layout), pypdf (text cleanup).  
* **Framework:** LangChain.

## **Configuration**

The application behavior is controlled via environment variables in docker-compose.yml:

| Variable | Default | Description |
| :---- | :---- | :---- |
| OLLAMA_HOST | http://ollama:11434 | Internal URL for the Ollama service. |
| EMBED_MODEL | nomic-embed-text | The model used to turn text into vectors. |
| LLM_MODEL | qwen:4b | The chat model used for generating answers. |
| PYTHONUNBUFFERED | 1 | Ensures logs are printed immediately to the Docker console. |

## **Getting Started**

### **Prerequisites**

1. Docker must be installed and running.

### **Installation & Running**

1. Clone the repository to your local machine.  
2. Build the image with Docker
3. Create a folder named pdfs/ in the root directory and put in your PDFs. Some clean example PDFs are provided.
4. Bring up the image with Docker Compose
4. Start the application: Open your browser and navigate to http://localhost:5000.

## **Project Structure**

* main.py: Contains the core logic for spatial extraction, table-to-section resolution, and the RAG pipeline.  
* templates/index.html: The single-page dashboard for chatting, viewing sources, and inspecting the database.  
* ingested_state.json: A state file tracking file hashes to manage sync operations.  
* chroma_data/: The directory where the persistent vector database is stored.  
* pdfs/: The target directory for document ingestion.



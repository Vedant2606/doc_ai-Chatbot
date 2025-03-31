# Document Chatbot

This project is a chatbot that answers questions based on document content using FAISS for retrieval.

## Features
- Loads and processes PDF documents.
- Converts text into vector embeddings.
- Stores and retrieves embeddings using FAISS.
- Provides a chatbot interface via Streamlit.

## Project Structure
- `connect_llm_memory.py`: Connects the LLM with FAISS-based retrieval.
- `doc_bot.py`: Streamlit-based chatbot interface.
- `llm_memory.py`: Handles document loading, text processing, and embedding.

## Installation
### Prerequisites
Python 3.8 or higher.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-doc-bot.git
   cd llm-doc-bot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

## Usage
### Process Documents
Run:
```bash
python llm_memory.py
```

### Start Chatbot
Run:
```bash
streamlit run doc_bot.py
```

## Technologies Used
- **LangChain**
- **Hugging Face Embeddings**
- **FAISS**
- **Streamlit**
- **Python**


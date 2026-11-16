# 📚 Multi-PDF RAG Chatbot

A document-based Question Answering application built using **LangChain, Streamlit, Google Gemini, Hugging Face Embeddings, and FAISS**.

Upload multiple PDF documents and ask questions across them. The application retrieves relevant information from the uploaded documents and generates context-aware answers using Google's Gemini model, along with source references.

## 🚀 Features

- **Multiple PDF Upload:** Upload and process multiple PDF files simultaneously.
- **PDF Text Extraction:** Extract text from PDF documents using PyPDFLoader.
- **Text Chunking:** Split extracted text into smaller chunks using Recursive Character Text Splitter.
- **Hugging Face Embeddings:** Convert text chunks into numerical vector representations using `all-MiniLM-L6-v2`.
- **FAISS Vector Database:** Store and search document embeddings efficiently.
- **Semantic Search:** Retrieve relevant document chunks based on user questions.
- **Gemini AI Integration:** Generate answers using Google's Gemini model.
- **Source References:** Display the original PDF filename, page number, and supporting text excerpts.
- **Chat Interface:** Interact with your uploaded documents through a Streamlit chat interface.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG application framework |
| Streamlit | Web application interface |
| Google Gemini | Large Language Model |
| Hugging Face | Text embedding model |
| FAISS | Vector similarity search |
| PyPDF | PDF text extraction |
| uv | Python package and environment management |

## 🏗️ Application Architecture

```text
              PDF Documents
                    |
                    v
              PDF Upload
                    |
                    v
              Text Extraction
                    |
                    v
              Text Splitting
                    |
                    v
          Hugging Face Embeddings
                    |
                    v
             FAISS Vector DB
                    |
                    |
User Question ------+
       |
       v
Query Embedding / Semantic Retrieval
       |
       v
Relevant Document Chunks
       |
       v
Context + User Question
       |
       v
        Google Gemini
       |
       v
     Generated Answer
       |
       v
   Source References
(File Name + Page Number)
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sanjeevicky0402/multi-pdf-rag-chatbot.git

cd multi-pdf-rag-chatbot
```

### 2. Install dependencies

This project uses `uv` for dependency management.

```bash
uv sync
```

### 3. Configure the Gemini API key

Create a `.env` file in the project root directory.

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Important:** Never upload your `.env` file or expose your API key publicly.

### 4. Run the application

```bash
uv run streamlit run app.py
```

The Streamlit application will open in your browser.

## 💬 How to Use

1. Launch the Streamlit application.
2. Upload one or more PDF documents.
3. Click **Process PDFs and Build Knowledge Base**.
4. Wait for text extraction, embedding generation, and FAISS indexing.
5. Enter your question in the chat interface.
6. View the generated answer.
7. Expand the source references to verify the original PDF filename, page number, and supporting content.

## 📌 Source Attribution

A key objective of this project is to make document-based answers traceable.

Each extracted PDF page retains its original filename and page number through the text-splitting and retrieval stages.

The application displays retrieved source excerpts so users can verify the information against the original documents.

## 🔮 Future Improvements

- Persistent FAISS vector database storage.
- OCR support for scanned PDF documents.
- Improved citation validation and inline source references.
- Conversation memory for contextual follow-up questions.
- Advanced retrieval techniques such as reranking.
- Support for additional document formats.
- Deployment using Streamlit Community Cloud.

## 👨‍💻 Author

**Sanjeevikumar**

GitHub: [@sanjeevicky0402](https://github.com/sanjeevicky0402)

---

⭐ If you find this project useful, consider giving it a star!

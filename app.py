
import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Chatbot")
st.write("Upload multiple PDFs and ask questions across all of them.")


# --------------------------------------------------
# 2. ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectors" not in st.session_state:
    st.session_state.vectors = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None


# --------------------------------------------------
# 4. INITIALIZE MODELS
# --------------------------------------------------

@st.cache_resource
def initialize_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


@st.cache_resource
def initialize_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Check your .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2,
        api_key=api_key
    )

    return llm


# --------------------------------------------------
# 5. PDF EXTRACTION
# --------------------------------------------------

def extract_pdf_documents(uploaded_files):

    all_documents = []

    for uploaded_file in uploaded_files:

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            for page_doc in pages:

                # Preserve original PDF filename
                page_doc.metadata["source"] = uploaded_file.name

                # Convert zero-based page to one-based page number
                page_index = int(
                    page_doc.metadata.get("page", 0)
                )

                page_doc.metadata["page_number"] = page_index + 1

            all_documents.extend(pages)

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return all_documents


# --------------------------------------------------
# 6. TEXT SPLITTING
# --------------------------------------------------

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# --------------------------------------------------
# 7. EMBEDDING GENERATION
# --------------------------------------------------

def generate_embeddings(chunks, embeddings):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embeddings.embed_documents(texts)

    return vectors


# --------------------------------------------------
# 8. CREATE FAISS VECTOR DATABASE
# --------------------------------------------------

def create_vectorstore(chunks, vectors, embeddings):

    text_embeddings = [
        (chunk.page_content, vector)
        for chunk, vector in zip(chunks, vectors)
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore


# --------------------------------------------------
# 9. RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

def retrieve_documents(question, vectorstore, k=5):

    retrieved_docs = vectorstore.similarity_search(
        question,
        k=k
    )

    return retrieved_docs


# --------------------------------------------------
# 10. GENERATE ANSWER WITH GEMINI
# --------------------------------------------------

def generate_answer(question, retrieved_docs, llm):

    context_parts = []
    source_map = {}

    for index, doc in enumerate(retrieved_docs, start=1):

        source_id = f"S{index}"

        filename = doc.metadata.get(
            "source",
            "Unknown file"
        )

        page_number = doc.metadata.get(
            "page_number",
            "Unknown"
        )

        source_map[source_id] = {
            "filename": filename,
            "page_number": page_number,
            "content": doc.page_content
        }

        context_parts.append(
            f"""
            [{source_id}]
            File: {filename}
            Page: {page_number}

            Content:
            {doc.page_content}
            """
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
    You are a document-based question answering assistant.

    Answer the user's question using ONLY the information
    available in the provided PDF context.

    STRICT RULES:

    1. Do not use outside knowledge.
    2. Do not invent facts or information.
    3. If the context does not contain enough information,
       clearly say:
       "The uploaded PDFs do not contain enough information
       to answer this question."
    4. Cite the source IDs that support each important claim.
    5. Use citations in this format: [S1], [S2].
    6. Never create or modify source IDs.
    7. If multiple PDFs contain relevant information,
       combine their information and cite the relevant sources.
    8. Give a clear and easy-to-understand answer.

    PDF CONTEXT:

    {context}

    USER QUESTION:

    {question}

    ANSWER:
    """

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            str(item.get("text", item))
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    return answer, source_map


# --------------------------------------------------
# 11. PDF UPLOAD INTERFACE
# --------------------------------------------------

st.sidebar.header("📂 Upload PDF Files")

uploaded_files = st.sidebar.file_uploader(
    "Choose your PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# 12. DETECT CHANGES IN UPLOADED FILES
# --------------------------------------------------

if uploaded_files:

    signature_hash = hashlib.sha256()

    for uploaded_file in uploaded_files:

        signature_hash.update(uploaded_file.name.encode())
        signature_hash.update(uploaded_file.getvalue())

    current_signature = signature_hash.hexdigest()

    if current_signature != st.session_state.file_signature:

        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.vectors = []
        st.session_state.vectorstore = None
        st.session_state.messages = []

        st.session_state.file_signature = current_signature

        st.info(
            "New PDF files detected. Process the PDFs to build "
            "the knowledge base."
        )

else:

    if st.session_state.file_signature is not None:

        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.vectors = []
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.file_signature = None


# --------------------------------------------------
# 13. DISPLAY UPLOADED FILES
# --------------------------------------------------

if uploaded_files:

    st.sidebar.subheader("Uploaded Files")

    for uploaded_file in uploaded_files:

        st.sidebar.write(
            f"📄 {uploaded_file.name}"
        )

    st.sidebar.write(
        f"Total PDFs: {len(uploaded_files)}"
    )


# --------------------------------------------------
# 14. PROCESS PDF BUTTON
# --------------------------------------------------

if uploaded_files:

    if st.button(
        "⚙️ Process PDFs and Build Knowledge Base",
        type="primary"
    ):

        try:

            with st.spinner("Loading AI models..."):

                embeddings = initialize_embeddings()
                llm = initialize_llm()

            with st.spinner("Extracting PDF text..."):

                documents = extract_pdf_documents(
                    uploaded_files
                )

                st.session_state.documents = documents

            if not documents:

                st.error(
                    "No pages were extracted from the uploaded PDFs."
                )

            else:

                with st.spinner("Splitting PDF text into chunks..."):

                    chunks = split_documents(documents)

                    st.session_state.chunks = chunks

                if not chunks:

                    st.error(
                        "No extractable text was found in these PDFs. "
                        "They may contain scanned pages requiring OCR."
                    )

                else:

                    with st.spinner(
                        "Generating Hugging Face embeddings..."
                    ):

                        vectors = generate_embeddings(
                            chunks,
                            embeddings
                        )

                        st.session_state.vectors = vectors

                    with st.spinner(
                        "Creating FAISS vector database..."
                    ):

                        vectorstore = create_vectorstore(
                            chunks,
                            vectors,
                            embeddings
                        )

                        st.session_state.vectorstore = vectorstore

                    st.success(
                        "PDF knowledge base created successfully!"
                    )

        except Exception as e:

            st.error(
                f"PDF processing failed: {str(e)}"
            )


# --------------------------------------------------
# 15. KNOWLEDGE BASE SUMMARY
# --------------------------------------------------

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("📊 Knowledge Base Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PDF Pages",
        len(st.session_state.documents)
    )

    col2.metric(
        "Text Chunks",
        len(st.session_state.chunks)
    )

    col3.metric(
        "Embedding Dimensions",
        len(st.session_state.vectors[0])
    )

    with st.expander("View processed PDF details"):

        for doc in st.session_state.documents:

            st.write(
                f"📄 {doc.metadata.get('source')} "
                f"— Page {doc.metadata.get('page_number')}"
            )

    with st.expander("Preview text chunks"):

        for index, chunk in enumerate(
            st.session_state.chunks[:5],
            start=1
        ):

            st.markdown(f"**Chunk {index}**")

            st.write(
                f"Source: {chunk.metadata.get('source')}"
            )

            st.write(
                f"Page: {chunk.metadata.get('page_number')}"
            )

            st.write(chunk.page_content)

            st.divider()


# --------------------------------------------------
# 16. CHAT INTERFACE
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Questions About Your PDFs")

if st.session_state.vectorstore is None:

    st.info(
        "Upload your PDFs and click "
        "'Process PDFs and Build Knowledge Base' "
        "before asking questions."
    )

else:

    # Display previous chat messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Accept user question
    user_question = st.chat_input(
        "Ask something about your uploaded PDFs..."
    )

    if user_question:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):

            st.markdown(user_question)

        # Retrieve and generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching PDFs and generating answer..."
            ):

                try:

                    embeddings = initialize_embeddings()
                    llm = initialize_llm()

                    # Retrieve relevant chunks
                    retrieved_docs = retrieve_documents(
                        user_question,
                        st.session_state.vectorstore,
                        k=5
                    )

                    # Generate grounded answer
                    answer, source_map = generate_answer(
                        user_question,
                        retrieved_docs,
                        llm
                    )

                    st.markdown(answer)

                    # Display source references
                    st.markdown("### 📚 Retrieved Sources")

                    for source_id, source in source_map.items():

                        with st.expander(
                            f"[{source_id}] "
                            f"{source['filename']} "
                            f"— Page {source['page_number']}"
                        ):

                            st.write(
                                source["content"]
                            )

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        f"Unable to generate answer: {str(e)}"
                    )


# --------------------------------------------------
# 17. CLEAR CHAT
# --------------------------------------------------

if st.session_state.messages:

    if st.sidebar.button("🗑️ Clear Chat History"):

        st.session_state.messages = []

        st.rerun()

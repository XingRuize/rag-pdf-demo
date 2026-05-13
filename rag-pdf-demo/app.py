import os
import tempfile
import hashlib
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)


def build_vector_store(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = uploaded_file.name

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma(
        collection_name=f"pdf_rag_{file_hash[:12]}",
        embedding_function=embeddings
    )

    vector_store.add_documents(
        documents=chunks,
        ids=[str(uuid4()) for _ in chunks]
    )

    os.remove(temp_file_path)

    return vector_store, len(documents), len(chunks), file_hash


def answer_question(vector_store, question):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ]
    )

    system_prompt = """
You are a careful PDF question-answering assistant.

Rules:
1. Answer only using the provided context.
2. If the context does not contain the answer, say: "I cannot find this information in the PDF."
3. Do not make up facts.
4. Be clear and concise.
5. When useful, mention which source chunk supports the answer.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
        temperature=0
    )

    response = llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt)
        ]
    )

    return response.content, relevant_docs


st.title("📄 PDF RAG Assistant")
st.write("Upload a PDF, ask questions, and get answers grounded in the document.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    current_file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.get("file_hash") != current_file_hash:
        with st.spinner("Reading PDF and building vector database..."):
            vector_store, page_count, chunk_count, file_hash = build_vector_store(uploaded_file)

            st.session_state["vector_store"] = vector_store
            st.session_state["file_hash"] = file_hash
            st.session_state["page_count"] = page_count
            st.session_state["chunk_count"] = chunk_count

        st.success("PDF processed successfully.")

    st.info(
        f"Loaded {st.session_state['page_count']} pages "
        f"and created {st.session_state['chunk_count']} chunks."
    )

    question = st.text_input(
        "Ask a question about the PDF",
        placeholder="Example: What is the late policy?"
    )

    if question:
        with st.spinner("Searching the PDF and generating answer..."):
            answer, sources = answer_question(
                st.session_state["vector_store"],
                question
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sources")

        for i, doc in enumerate(sources):
            page = doc.metadata.get("page", "Unknown")
            page_display = page + 1 if isinstance(page, int) else page

            with st.expander(f"Source {i + 1} — Page {page_display}"):
                st.write(doc.page_content)
else:
    st.warning("Upload a PDF to begin.")
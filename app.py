import streamlit as st
import os, pathlib
from ingest import extract_text_from_pdf, extract_text_from_html, extract_text_from_image, chunk_text
from embed import Embedder
from rag import SimpleRAG

st.set_page_config(page_title="Local/ChatGPT RAG", layout="wide")
st.title("RAG System with Local LLM or ChatGPT")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource
def get_embedder():
    return Embedder()

embedder = get_embedder()

# -----------------------------
# Ingestion
# -----------------------------
st.sidebar.header("Upload Files")
uploaded = st.sidebar.file_uploader(
    "PDF / HTML / Images",
    accept_multiple_files=True,
    type=["pdf", "html", "htm", "png", "jpg", "jpeg"]
)

if st.sidebar.button("Ingest Files"):
    docs, metas = [], []
    for uf in uploaded:
        fp = os.path.join(DATA_DIR, uf.name)
        with open(fp, "wb") as f:
            f.write(uf.getbuffer())

        ext = pathlib.Path(fp).suffix.lower()

        if ext == ".pdf":
            text = extract_text_from_pdf(fp)
        elif ext in [".html", ".htm"]:
            text = extract_text_from_html(fp)
        elif ext in [".png", ".jpg", ".jpeg"]:
            text = extract_text_from_image(fp)
        else:
            text = ""

        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            docs.append(ch)
            metas.append({"source": uf.name, "chunk": i})

    if docs:
        rag = SimpleRAG(
            index_path=os.path.join(DATA_DIR, "faiss_index.bin"),
            meta_path=os.path.join(DATA_DIR, "vectors.pkl"),
            embedder=embedder
        )
        rag.build(docs, metas)
        st.sidebar.success(f"Indexed {len(docs)} chunks.")
    else:
        st.sidebar.error("No extractable text found.")

# -----------------------------
# Query area
# -----------------------------
st.sidebar.header("Query")
query = st.sidebar.text_input("Enter your question")
top_k = st.sidebar.slider("Top K", 1, 10, 5)

llm_provider = st.sidebar.selectbox("Choose LLM Provider", ["OpenAI ChatGPT", "Local HF Model"])

if llm_provider == "OpenAI ChatGPT":
    from llm_openai import ChatGPT_LLM
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model_name = st.sidebar.text_input("Model", value="gpt-4.1-mini")
else:
    from llm_backend import LocalLLM
    model_name = st.sidebar.text_input("HF Model", value="google/flan-t5-small")
    device = st.sidebar.selectbox("Device", ["cpu", "cuda"])

if st.sidebar.button("Search"):
    if not query:
        st.error("Please enter a question.")
    else:
        rag = SimpleRAG(
            index_path=os.path.join(DATA_DIR, "faiss_index.bin"),
            meta_path=os.path.join(DATA_DIR, "vectors.pkl"),
            embedder=embedder
        )
        results = rag.query(query, top_k=top_k)

        st.write("### Retrieved Chunks")
        contexts = []
        for r in results:
            st.write(f"**Score:** {r['score']:.4f} — {r['metadata']}")
            contexts.append(f"{r['metadata']}")

        st.write("### Answer")

        if llm_provider == "OpenAI ChatGPT":
            if not api_key:
                st.error("Please enter your OpenAI API key.")
            else:
                llm = ChatGPT_LLM(api_key=api_key, model=model_name)
                answer = llm.generate(query, contexts)
                st.write(answer)

        else:
            llm = LocalLLM(model_name=model_name, device=device)
            answer = llm.generate(query, contexts)
            st.write(answer)

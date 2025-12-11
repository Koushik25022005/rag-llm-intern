import streamlit as st
import os, tempfile, pathlib, pickle, json
from ingest import extract_text_from_pdf, extract_text_from_html, extract_text_from_image, chunk_text
from embed import Embedder
from rag import SimpleRAG
from llm_backend import LocalLLM

st.set_page_config(page_title='Local LLM-RAG', layout='wide')
st.title('Local LLM-RAG — Streamlit demo (local)')

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource
def get_embedder():
    return Embedder()

@st.cache_resource
def get_rag():
    return SimpleRAG()

embedder = get_embedder()
rag = get_rag()

st.sidebar.header('Ingest files')
uploaded = st.sidebar.file_uploader('Upload PDF / HTML / Image', accept_multiple_files=True, type=['pdf','html','htm','png','jpg','jpeg'])

if st.sidebar.button('Ingest uploaded'):
    if not uploaded:
        st.sidebar.warning('Upload one or more files first.')
    else:
        docs = []
        metas = []
        for uf in uploaded:
            name = uf.name
            ext = pathlib.Path(name).suffix.lower()
            fp = os.path.join(DATA_DIR, name)
            with open(fp, 'wb') as f:
                f.write(uf.getbuffer())
            if ext == '.pdf':
                text = extract_text_from_pdf(fp)
            elif ext in ['.html','.htm']:
                text = extract_text_from_html(fp)
            elif ext in ['.png','.jpg','.jpeg']:
                text = extract_text_from_image(fp)
            else:
                text = ''
            chunks = chunk_text(text)
            for i,ch in enumerate(chunks):
                docs.append(ch)
                metas.append({'source': name, 'chunk': i})
        if docs:
            from rag import SimpleRAG
            rag = SimpleRAG(index_path=os.path.join(DATA_DIR,'faiss_index.bin'),
                            meta_path=os.path.join(DATA_DIR,'vectors.pkl'),
                            embedder=embedder)
            rag.build(docs, metas)
            st.sidebar.success(f'Indexed {len(docs)} chunks from {len(uploaded)} files.')
        else:
            st.sidebar.warning('No text extracted from uploaded files.')

st.sidebar.header('Query')
query = st.sidebar.text_input('Ask a question')
top_k = st.sidebar.slider('Top K', 1, 10, 5)
model_name = st.sidebar.text_input('HF model for generation', value='google/flan-t5-small')
device = st.sidebar.selectbox('Device', ['cpu','cuda']) if 'cuda' else 'cpu'

if st.sidebar.button('Search'):
    if not query:
        st.sidebar.warning('Enter a question in the Query box.')
    else:
        rag = SimpleRAG(index_path=os.path.join(DATA_DIR,'faiss_index.bin'),
                        meta_path=os.path.join(DATA_DIR,'vectors.pkl'),
                        embedder=embedder)
        res = rag.query(query, top_k=top_k)
        st.write('### Top matches from vector DB')
        contexts = []
        for r in res:
            md = r['metadata']
            st.write(f"**score:** {r['score']:.4f} — **source:** {md.get('source')} chunk {md.get('chunk')}")
            contexts.append(f"source: {md.get('source')} chunk: {md.get('chunk')}")
        st.write('### Answer (from local LLM)')
        with st.spinner('Running local LLM...'):
            llm = LocalLLM(model_name=model_name, device=device)
            answer = llm.generate(query, contexts)
            st.markdown(answer)

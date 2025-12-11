import os, pickle, faiss, numpy as np
from typing import List, Tuple
from embed import Embedder

class SimpleRAG:
    def __init__(self, index_path='faiss_index.bin', meta_path='vectors.pkl', embedder=None):
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedder = embedder or Embedder()
        self.index = None
        self.metadatas = []
        self.dim = None
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load()

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path,'rb') as f:
            data = pickle.load(f)
            self.metadatas = data['metadatas']
            self.dim = data['dim']

    def build(self, docs: List[str], metadatas: List[dict]):
        embs = self.embedder.embed_documents(docs)
        self.dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.metadatas = metadatas
        self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path,'wb') as f:
            pickle.dump({'metadatas': self.metadatas, 'dim': self.dim}, f)

    def query(self, q: str, top_k=5):
        q_emb = self.embedder.embed_query(q).reshape(1,-1)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            results.append({'score': float(score), 'metadata': self.metadatas[idx]})
        return results

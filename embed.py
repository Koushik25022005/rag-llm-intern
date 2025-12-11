import os, pickle
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, docs):
        return self.model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    def embed_query(self, q):
        return self.model.encode([q], convert_to_numpy=True)[0]

import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config

class EnhancedVectorStore:
    def __init__(self):
        self.model = SentenceTransformer(config.embedding_model)
        self.index = None
        self.meta = []

    def build_index(self, all_chunks, sources):
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        self.meta = [{"text": chunk, "source": src} for chunk, src in zip(all_chunks, sources)]
        os.makedirs("index", exist_ok=True)
        faiss.write_index(self.index, config.index_path)
        with open(config.chunk_meta_path, "w") as f:
            json.dump(self.meta, f)

    def load_index(self):
        self.index = faiss.read_index(config.index_path)
        with open(config.chunk_meta_path) as f:
            self.meta = json.load(f)

    def search(self, query, k=5):
        emb = self.model.encode([query])
        distances, indices = self.index.search(np.array(emb), k)
        return [self.meta[i] for i in indices[0]]

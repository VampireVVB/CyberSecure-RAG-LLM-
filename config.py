import os
class ConfigManager:
    def __init__(self):
        self.api_base_url = "http://localhost:11434/v1"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.index_path = "index/faiss.index"
        self.chunk_meta_path = "index/chunk_metadata.json"
        self.chat_history_file = "conversation_history.json"
        self.chunk_size = 500

config = ConfigManager()


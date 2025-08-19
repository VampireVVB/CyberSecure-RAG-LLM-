from vector_store import EnhancedVectorStore
from llm_interface import LLaMAGenerator
from config import config
import json, os
from datetime import datetime

class ImprovedRAGWithQwen14B:
    def __init__(self):
        self.vector_store = EnhancedVectorStore()
        self.vector_store.load_index()
        self.llm = LLaMAGenerator()

    def ask(self, query):
        chunks = self.vector_store.search(query, k=3)
        context = ""
        for i, c in enumerate(chunks):
            context_piece = f"[{i+1}] {c['text']} (Source: {c['source']})\n"
            if len(context + context_piece) > 6000:  
                break
            context += context_piece
        prompt = f"Answer using this context:\n{context}\n\nQ: {query}\nA:"
        response = self.llm.generate(prompt)
        sources = [c['source'] for c in chunks]
        self.save_response(query, response, sources)
        return response, sources

    def save_response(self, query, response, sources):
        os.makedirs("responses", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(response) > 1000:
            with open(f"responses/response_{timestamp}.txt", "w") as f:
                f.write(response)
        if os.path.exists(config.chat_history_file):
            with open(config.chat_history_file) as f:
                history = json.load(f)
        else:
            history = []
        history.append({"query": query, "response": response, "sources": sources})
        with open(config.chat_history_file, "w") as f:
            json.dump(history, f, indent=2)

from processor import OptimizedPDFProcessor, AdaptiveChunker
from vector_store import EnhancedVectorStore
import os

processor = OptimizedPDFProcessor()
chunker = AdaptiveChunker()
store = EnhancedVectorStore()

all_chunks, all_sources = [], []
pdf_dir = "pdfs"

for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_dir, file)
        text = processor.extract_text(path)
        chunks = chunker.chunk(text)
        all_chunks.extend(chunks)
        all_sources.extend([file]*len(chunks))

store.build_index(all_chunks, all_sources)
print("✅ Indexing complete.")

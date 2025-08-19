import fitz
import nltk
nltk.download('punkt')

class OptimizedPDFProcessor:
    def extract_text(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            return "\n".join([page.get_text() for page in doc])
        except:
            return ""

class AdaptiveChunker:
    def chunk(self, text, max_tokens=500, overlap=50):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current = []
        count = 0
        for sent in sentences:
            count += len(sent.split())
            current.append(sent)
            if count >= max_tokens:
                chunks.append(" ".join(current))
                current = current[-overlap:]
                count = len(" ".join(current).split())
        if current:
            chunks.append(" ".join(current))
        return chunks

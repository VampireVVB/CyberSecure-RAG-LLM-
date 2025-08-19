import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_model import ImprovedRAGWithQwen14B

def main():
    print("🔐 CyberSecure-Learn | Secure Coding Tutor CLI")
    print("Type 'exit' to quit.\n")
    rag = ImprovedRAGWithQwen14B()
    while True:
        query = input("🧑‍💻 You: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        response, sources = rag.ask(query)
        print(f"\n🤖 Tutor:\n{response}\n")
        print(f"📚 Sources: {', '.join(set(sources))}\n")

if __name__ == "__main__":
    main()

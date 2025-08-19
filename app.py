from flask import Flask, render_template, request, jsonify
from rag_model import ImprovedRAGWithQwen14B

app = Flask(__name__)
rag = ImprovedRAGWithQwen14B()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    answer, sources = rag.ask(query)
    return jsonify({
        "answer": answer,
        "sources": list(set(sources))
    })

if __name__ == '__main__':
    app.run(debug=True)

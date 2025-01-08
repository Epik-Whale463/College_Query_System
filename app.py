from flask import Flask, request, jsonify, render_template
from rag_functions import initialize_rag, query_rag
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

file_paths = {
    'txt': './data/faculty.txt',
    'txt2': './data/marks.txt',
}

initialize_rag(file_paths)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query/", methods=["POST"])
def query_endpoint():
    try:
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query parameter is missing"}), 400
        result = query_rag(query)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a health check endpoint
@app.route("/healthz")
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 轮询端口
ports = [5005, 5006]
current = 0

@app.route("/minigpt4/answer", methods=["POST"])
def forward():
    global current
    port = ports[current]
    current = (current + 1) % len(ports)
    try:
        resp = requests.post(f"http://127.0.0.1:{port}/minigpt4/answer", json=request.get_json())
        return (resp.content, resp.status_code, resp.headers.items())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
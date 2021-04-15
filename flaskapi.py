from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=[ 'POST'])
async def hello_world():
    if request.method == 'POST':
        return 'Hello, World!'


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)

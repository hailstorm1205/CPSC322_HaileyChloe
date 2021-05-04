from flask import Flask, jsonify, request
import pickle
import os

# Make a Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Hello World!</h1>", 200

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000) # Heroku will set the PORT environment variable for web traffic
    app.run(debug=False, host="0.0.0.0", port=port) # TODO: set debug=False before deployment!
from flask import Flask
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Hello World!</h1>"

if __name__ == "__main__":
    app.run(debug=True) #TODO: set debug=False before deployment!
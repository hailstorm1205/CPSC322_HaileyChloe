"""
Programmer: Hailey Mueller
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 5/04/21
Bonus?: TBD

Description: This file creates a flask application file for our final project.
"""

from flask import Flask, jsonify, request
import pickle
import os

# Make a Flask app
app = Flask(__name__)

# Homepage
@app.route("/", methods=["GET"])
def index():
    return "<h1>Welcome to our Wine Classification!</h1>", 200

# one for the /predict 
@app.route("/predict", methods=["GET"])
def predict():
    # use the request.args dictionary
    fixed_acidity = request.args.get("fixed acidity", "")
    volatile_acidity = request.args.get("volatile acidity","")
    citric_acid = request.args.get("citric acid","")
    residual_sugar = request.args.get("residual sugar","")
    chlorides = request.args.get("chlorides","")
    free_sulfur = request.args.get("free sulfur dioxide","")
    total_sulfur = request.args.get("total sulfur dioxide","")
    density = request.args.get("density","")
    pH = request.args.get("pH","")
    sulphates = request.args.get("sulphates","")
    alcohol = request.args.get("alcohol","")
    #print("fixed acidity:", fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol)

    prediction = predict_well([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol]) #[level, lang, tweets, phd])
    # if anything goes wrong, predict_well() is going to return None
    #return jsonify(prediction)
    if prediction is not None:
        result = {"wine quality": prediction[0]}
        return jsonify(result), 200
    else: 
        # failure!!
        return "Error making prediction", 400

def predict_well(instance):
    infile = open("tree.p", "rb")
    nb_instance = pickle.load(infile)
    infile.close()

    try: 
        return nb_instance.predict(instance)
    except:
        return None

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000) # Heroku will set the PORT environment variable for web traffic
    app.run(debug=False, host="0.0.0.0", port=port) # TODO: set debug=False before deployment!
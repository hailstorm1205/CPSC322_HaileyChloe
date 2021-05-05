"""
Programmer: Hailey Mueller
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 5/03/21
Bonus?: TBD

Description: This file creates a flask application for our final project.
"""

from flask import Flask, jsonify, request
import pickle
import os

# Make a Flask app
app = Flask(__name__)

# Homepage
@app.route("/", methods=["GET"])
def index():
    return "<h1>Hello World!</h1>", 200

@app.route("/test", methods=["GET"])
def test():
    return "<h1>TESTING</h1>", 300

# one for the /predict 
@app.route("/predict", methods=["GET"])
def predict():
    # use the request.args dictionary
    level = request.args.get("level", "")
    lang = request.args.get("lang", "")
    tweets = request.args.get("tweets", "")
    phd = request.args.get("phd", "")
    print("level:", level, lang, tweets, phd)

    '''fixed_acidity = request.args.get("fixed acidity", "")
    volatile_acidity = request.args.get("volatile acidity","")
    citric_acid = request.args.get("citric acid","")
    residual_sugar = request.args.get("residual sugar","")
    chlorides = request.args.get("chlorides","")
    free_sulfur = request.args.get("free sulfur dioxide","")
    total_sulfur = request.args.get("total sulfur dioxide","")
    density = request.args.get("density","")
    pH = request.args.get("pH","")
    sulphates = request.args.get("sulphates","")
    alcohol = request.args.get("alcohol","")'''
    #print("level:", level, lang, tweets, phd)
    # task: extract the remaining 3 args

    # get a prediction for this unseen instance via the tree
    # return the prediction as a JSON response

    prediction = predict_well([level, lang, tweets, phd])
    # if anything goes wrong, predict_well() is going to return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else: 
        # failure!!
        return "Error making prediction", 400

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        # now I need to find which "edge" to follow recursively
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match!! recurse!!
                return tdidt_predict(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label

def predict_well(instance):
    # 1. we need to a tree (and its header)
    # we need to save a trained model (fit()) to a file
    # so we can load that file into memory in another python
    # process as a python object (predict())
    # import pickle and "load" the header and interview tree 
    # as Python objects we can use for step 2
    infile = open("tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    print("header:", header)
    print("tree:", tree)

    # 2. use the tree to make a prediction
    try: 
        return tdidt_predict(header, tree, instance) # recursive function
    except:
        return None

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000) # Heroku will set the PORT environment variable for web traffic
    app.run(debug=False, host="0.0.0.0", port=port) # TODO: set debug=False before deployment!
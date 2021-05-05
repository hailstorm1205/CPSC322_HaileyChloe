"""
Programmer: Hailey Mueller
Class: CPSC 322-01, Spring 2021
Assignment: Final Project
Date Last Updated: 5/04/21
Bonus?: TBD

Description: This file creates a flask predictor file for our final project.
"""

import requests # lib to make http requests
import json # lib to help with parsing JSON objects

#Create url
url = "http://127.0.0.0:5000/predict?fixed_acidity=2&volatile_acidity=1&citric_acid=4&residual_sugar=5&chlorides=2&free_sulfur=1&total_sulfur=1&density=6&pH=3&sulphates=6&alcohol=4"


# make a GET request to get the search results back
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
response = requests.get(url=url)

# first thing... check the response status code 
status_code = response.status_code
print("status code:", status_code)

if status_code == 200:
    # success! grab the message body
    json_object = json.loads(response.text)
    print(json_object)
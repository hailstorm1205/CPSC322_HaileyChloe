import requests # lib to make http requests
import json # lib to help with parsing JSON objects

# url = "https://interview-flask-app.herokuapp.com/predict?level=Junior&lang=Java&tweets=yes&phd=yes"
url = "http://127.0.0.0:5000/predict?level=Junior&lang=Java&tweets=yes&phd=yes"
#url = "https://cpsc322-finalproject2.herokuapp.com/predict?level=Junior&lang=Java&tweets=yes&phd=yes"


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
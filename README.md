# Wine Classification

This repository holds the files and classification tools necessary in order to classify the quality of red wine
based on 12 given attributes.

## Attributes

- Fixed acidity (int)
- Volatile acidity (int)
- Citric acid (int)
- Residual sugar (int)
- Chlorides (int)
- Free sulfur dioxide (int)
- Total sulfur dioxide (int)
- Density (int)
- pH (int)
- Sulphates (int)
- Alcohol (int)
- Quality* (int)

## Installation

In order to run the flask application locally on your machine, you will need to run the "flaskApp.py" file.

```bash
python3 flaskApp.py
```

This command will start running the file on port 5000 of your local machine. Then, you can change the 11 parameters
to your liking in order to predict the quality of the wine.

## Organization

Our repository is organized accordingly:

The folder "input_files" holds the csv file that holds the data we collected about red wine including the scientific
attributes associated with determining the quality of the wine.

The folder "mysklearn" holds all the necessary code used to classify a given dataset.

The files not organized include our Jupyter notebooks and the files necessary for web deployment using Flask and Heroku.
Those files include "Dockerfile", "flaskApp.py", "flaskPickler.py", "flaskPredicter.py", "heroku.yml", "Procfile", "requirements.txt",
and "tree.p".
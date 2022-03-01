# Employee Attrition Analysis: 

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Installation](#installation)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Directory Tree](#directory-tree)
  * [Future scope of project](#future-scope)


## Demo
### Link: 
[https://employee-attrition-analysis.herokuapp.com/](https://employee-attrition-analysis.herokuapp.com/)

### Website screenshot:
[![](/images/webpage_screencapture1.png)](https://employee-attrition-analysis.herokuapp.com/)

### Screenshot of the results:
[![](/images/webpage_screencapture2.PNG)](https://employee-attrition-analysis.herokuapp.com/)

## Overview
This data science project walks through step by step process of building HR-Department webpage to help them to predict if their employees may leave the company or not, and if yes, predict after how many years they will leave it.

We will first build 2 models using sklearn, logistic regression, and linear regression using (HR Analytics Case Study) dataset from [kaggle](https://www.kaggle.com/vjchoudhary7/hr-analytics-case-study).

Second step would be to write a python flask server that uses the saved model to serve http requests.

Third component is the website built in html and css that allows user to enter general employee info., employee survey results, and manager survey results and it will call python flask server to retrieve the predicted attrition results.

This project covers almost all data science concepts such as data load and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, gridsearchcv for hyperparameter tunning, k fold cross validation etc. 

[![](/images/Workflow-chart.JPG )](https://employee-attrition-analysis.herokuapp.com/)

Technology and tools wise this project covers:
- Python
- Numpy and Pandas for data cleaning
- Plotly, Seaborn, and Matplotlib for data visualization
- Sklearn for model building.
- Jupyter notebook, Visual Studio code and pycharm as IDE
- Python flask for http server
- HTML/CSS for UI
- Heroku cloud platform as a service for deploying the project.

## Installation
The Code is written in Python 3.10.2. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Deployement on Heroku
Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.

[![](https://i.imgur.com/dKmlpqX.png)](https://heroku.com)

Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```
├── static 
│   ├── CSS
│   	├── Styles.css
├── saved_models 
│   ├── GradientBoostingRegressor.pkl
│   ├── RandomForestClassifier.pkl
├── training_code 
│   ├── employees_attrition.ipynb
│   ├── employees_attrition.py
├── template
│   ├── home.html
├── images
│   ├── bg.png
│   ├── Workflow-chart.JPG
│   ├── webpage_screencapture1.png
│   ├── webpage_screencapture2.png
├── Procfile
├── README.md
├── app.py
├── requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/) 

## Future Scope

* Optimize Flask app.py
* Front-End 
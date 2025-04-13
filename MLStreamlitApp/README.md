Content Suggestions:
Instructions: Include step-by-step instructions on how to run the app locally and a link to the deployed version. List all necessary libraries and versions.
App Features: Explain the models you used and describe how hyperparameters are selected or tuned.
References: Include links to documentation, tutorials, or guides that informed your implementation.
Visual Examples: Consider adding screenshots of the app or charts produced by the app.

# Machine Learning Streamlit App 

## 📕 Project Overview 
#### This interactive Streamlit app allows users to use 3 different models (decision trees, linear regression and logistic regression) to evaluate data. 3 sample datasets are loaded, but users also have the option to upload their own dataset as a CSV. Users can view various metrics including F1 scores, RMSE and R squared scores, Gini index and entropy, depending on what model is selected. 
- **Project Goal:** Demonstrate the ability to create an interactive Streamlit app that showcases the pros and cons of different machine learning models for evaluation and prediction. 
- **Skills and Packages Applied:** Python, Streamlit, Pandas, Numpy, Seaborn, Matplotlib, Scikit Learn, Graphviz, Decision Trees, Logistic Regression, and Linear Regression

## 📖 Instructions 
This script was written in Visual Studio code. The instructions below detail how to run the code using this platform.

**Run the App Locally:**



**1.** Import the packages and libraries detailed in the "Necessary Libraries and Versions section" below.


**2.** Run Streamlit in your terminal and open the prompted link to view the app. For more detailed instructions on this setup, click [here](https://docs.kanaries.net/topics/Streamlit/streamlit-vscode)


**3.** Once these steps are completed, you will be able to run the code in your own VS code environment! 

**Direct Link to the Deployed Version**
To view the App without running the code, click here 



### 📚 Necessary Libraries and Versions
- **necessary libraries and packages:** Streamlit, Pandas, Numpy, Seaborn, Matplotlib, Scikit Learn, and Graphviz
 - For more specifics, view the code below:
````
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_curve, roc_auc_score
import graphviz
````
- ***Note:*** Graphviz must also be Conda installed in Visual Studio code. For installation instructions, click [here](https://anaconda.org/conda-forge/python-graphviz) 

## 📲 App Features  
****The features 3 model types: Decision Trees, Logisitic Regression, and Linear Regression. A brief description of each is provided below, along with details about the hyperparameters and evaluation metrics available in the app.****
### Decision Trees
- Can be used for classification and regression. In simple terms, a decision tree outlines options based on whether a binary variable is true or false. Each answer leads to a lower level where the process is repeated, until a conclusion (or prediction in the case of regression) can be reached, represented by a leaf node, something with no more branches
**Hyperparameters:** The hyperparameters used in this Streamlit are depicted and described below. 
<img width="794" alt="Screenshot 2025-04-13 at 11 20 46 AM" src="https://github.com/user-attachments/assets/9f2da996-c7cf-49ab-9448-aa60cefbbb9c" />


- ***Maximum depth:*** Changes how many levels the decision tree will have, with a smaller number meaning there will be fewer levels.

- ***Minimum samples to split:*** The minimum number of samples required to allow a split in node. This is a helpful hyperparameter to help prevent oversplitting,when the data is split too much so the leaves will not have enough samples.
- ***Entropy vs Gini Index vs Log loss:***
  - ***Gini index:*** Measure of diversity based on the probability of certain values being in a node. A lower score, closer to 0, indicates a purer and better split.
  - Lower gini index = better split 

- ***Entropy:*** Measures the diversity of the set, if the data split results in a lower combined entropy, it’s a better split. 
  - Lower average entropy = better split 

- ***Log loss:*** An evaluation metric for binary classification models
  - Lower log loss = predicted probabilities are more accurate in the model 

#### Linear Regression 
- Can be used for linear relationships where the feature variables are numeric or categorical and the target is something on a continuous numeric scale. A linear regression model is helpful for evaluating a relationship between variables.


#### Logistic Regression 
- **Logistic Regression:** Can be used with binary variables to evaluate how these features influence the probability of something happening. A logisitc regression may be the best option if the goal is to see how each selected feature impact the outcome's probability.
- 
## ℹ️ References 
[Installing Graphviz](https://anaconda.org/conda-forge/python-graphviz)
[Logisitic Regression Inspiration](https://github.com/ragh945/tuning-of-hyperparameters-for-logisticregression_using-decision-surfaces/blob/main/Hyperparameters.py)
[Linear Regression Inspiration](https://github.com/matejve/linear_regression_demo/blob/main/Introduction_Page.py)
[Other Streamlit App Inspiration](https://varunlobo-decision-tree-using-streamlit-main-myvzpw.streamlit.app/)


**Learn More About the Model Types:**
[Grokking Machine Learning](https://www.google.com/books/edition/Grokking_Machine_Learning/X-9MEAAAQBAJ?hl=en&gbpv=1&pg=PA1&printsec=frontcover) 


## 📸 Visual Examples: 

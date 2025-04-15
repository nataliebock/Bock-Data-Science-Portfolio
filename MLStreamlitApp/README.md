# Machine Learning Streamlit App 

## 📕 Project Overview 
#### This interactive Streamlit app allows users to use 3 different models (decision trees, linear regression and logistic regression) to evaluate data. 3 sample datasets are loaded, but users also have the option to upload their own dataset as a CSV. Users can evaluate model performance by viewing various [metrics](#-App-Features) depending on what model is selected. 
- **Project Goal:** Demonstrate the ability to create an interactive Streamlit app that showcases the pros and cons of different machine learning models for evaluation and prediction. 
- **Skills and Packages Applied:** Python, Streamlit, Pandas, Numpy, Seaborn, Matplotlib, Scikit Learn, Graphviz, Machine Learning, Decision Trees, Logistic Regression, and Linear Regression

## 📖 Instructions 
This script was written in Visual Studio code. The instructions below detail how to run the code using this platform.

### 🏃 Run the App Locally: 



**1.** Import the packages and libraries detailed in the "Necessary Libraries and Versions section" below.


**2.** Run Streamlit in your terminal and open the prompted link to view the app. For more detailed instructions on this setup, click [here](https://docs.kanaries.net/topics/Streamlit/streamlit-vscode)


**3.** Once these steps are completed, you will be able to run the code in your own VS code environment! 

### 🔗 Direct Link to the Deployed Version 
To view the App without running the code, click [here](https://machinelearningappnkb.streamlit.app)


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
from sklearn.preprocessing import StandardScaler
````
- ***Note:*** Graphviz must also be Conda installed in Visual Studio code. For installation instructions, click [here](https://anaconda.org/conda-forge/python-graphviz) 

#### Versions 
Below are the listed versions of the packages and libraries used in this app:
````
graphviz==0.20.3
matplotlib==3.10.1
numpy==2.2.4
pandas==2.2.3
scikit_learn==1.6.1
seaborn==0.13.2
streamlit==1.37.1
````
## 📲 App Features  
****The app features 3 model types: Decision Trees, Logistic Regression, and Linear Regression. A brief description of each is provided below, along with details about the hyperparameters and evaluation metrics available in the app.****
### Decision Trees
- **Purpose:** Can be used for classification and regression.In simple terms, a decision tree outlines options based on whether a binary variable is true or false. Each answer leads to a lower level where the process is repeated, until a conclusion can be reached (or prediction in the case of regression), represented by a leaf node, a box with no branches. This means that no more splits can occur in the tree. This decision tree model used is for classification, and is therefore good if the target variable is categorical.
  
**Hyperparameters:** The hyperparameters used in this Streamlit are depicted and described below. 
<img width="794" alt="Screenshot 2025-04-13 at 11 20 46 AM" src="https://github.com/user-attachments/assets/9f2da996-c7cf-49ab-9448-aa60cefbbb9c" />


- ***Maximum depth:*** Changes how many levels the decision tree will have, with a smaller number meaning there will be fewer levels.

- ***Minimum samples to split:*** The minimum number of samples required to allow a split in node. This is a helpful hyperparameter to help prevent oversplitting, when the data is split too much so the leaves do not have enough samples.
- ***Entropy vs Gini Index vs Log loss:***
  - ***Gini index:*** Measure of diversity based on the probability of certain values being in a node. A lower score, closer to 0, indicates a purer and better split. <ins>Lower gini index = better split</ins> 

  - ***Entropy:*** Measures the diversity of the set, if the data split results in a lower combined entropy, it’s a better split. <ins>Lower average entropy = better split</ins> 

  - ***Log loss:*** An evaluation metric for binary classification models. <ins>Lower log loss = predicted probabilities are more accurate in the model</ins>

### Linear Regression 
- **Purpose:**  Can be used for linear relationships where the target is something on a continuous numeric scale, or a categorical numerical scale in some cases. A linear regression model is helpful for evaluating a relationship between variables, for example, a one unit increase in y increases x by 15 points. A good model choice is the target variable is numeric.
- **Scaled vs Unscaled:** The linear regression model also allows users to select whether they would like to use scaled or unscaled data. Using unscaled data can make it difficult to compare features measured in different units. By scaling the data, it becomes easier to draw comparisons across the different features.
- **Evaluation Metrics:** 
  - ***Mean Squared Error (MSE):*** Shows the average squared difference between actual and predicted values, where a lower value generally means a better model
  - ***Root Mean Squared Error (RMSE):*** The difference between actual and predicted values, in the same units as the target, a lower value generally means a better model
  - ***R² Score:*** Shows the proportion of variance that the model accounts for. On a scale from 0 - 1, a score closer to 1 indicates a model that better accounts for variance
  
**The scale prompt in Streamlit is shown below ⬇️**

<img width="339" alt="Screenshot 2025-04-14 at 6 48 36 PM" src="https://github.com/user-attachments/assets/7a1a450c-7073-4593-9845-b3398efab6ed" />


### Logistic Regression 
- **Purpose:** To evaluate the probability of the features influencing the target variable. The target variable should be categorical, while the features can be categorical or numeric. A good model selection if the intended target variable is binary.
- **Evaluation metrics:**
  - ***Classification report:*** displaying accuracy, precision, recall, and F1-score
  - ***Confusion matrix:*** Evaluates performance by comparing predictions from the model to actual values. The key parts of the confusion matrix are the true negatives (located in the top left corner), and the true positives (located in the bottom right corner). These show the number of times the model correctly predicts something that is actually negative or actually positive.
  - ***ROC curve:*** The ROC, or Reciever Operating Characteristic, Curve shows a plot of the True Positive Rate and the False Positive Rate. This is helpful for selecting the threshold, where the number of true positives can be maximized, while also keeping a low percentage of false positives.
  - ***The AUC Score:*** The AUC, or area under the curve, summarizes the model performance. The AUC score is on a scale from 0-1, and should be better than 0.5, which represents a 50 percent chance of correct prediction. If the score is better than 0.5, it means that the model is better at predicting true positives than random guessing.

   
## ℹ️ References 
[Installing Graphviz](https://anaconda.org/conda-forge/python-graphviz)
[Logisitic Regression Inspiration](https://github.com/ragh945/tuning-of-hyperparameters-for-logisticregression_using-decision-surfaces/blob/main/Hyperparameters.py)
[Linear Regression Inspiration](https://github.com/matejve/linear_regression_demo/blob/main/Introduction_Page.py)
[Other Streamlit App Inspiration](https://varunlobo-decision-tree-using-streamlit-main-myvzpw.streamlit.app/)
[Links to different Readme sections](https://gist.github.com/rachelhyman/b1f109155c9dafffe618)

**Learn More About the Sample Datasets**
[Penguins Sample Dataset](https://github.com/allisonhorst/palmerpenguins)
[Titanic Sample Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
[Iris Sample Dataset](https://www.kaggle.com/datasets/uciml/iris)

**Learn More About the Model Types:**
[Grokking Machine Learning](https://www.google.com/books/edition/Grokking_Machine_Learning/X-9MEAAAQBAJ?hl=en&gbpv=1&pg=PA1&printsec=frontcover) 


## 📸 Visual Examples 
- **Below is a decision tree made with the Titanic dataset, where a maximum depth of 4 and a minimum samples to split of 5 were selected, along with a gini criteria.** 
<img width="1434" alt="Screenshot 2025-04-13 at 1 52 32 PM" src="https://github.com/user-attachments/assets/a83fe648-6780-4729-bc5e-19c700fe52c5" />

- **Below is a visualization of a confusion matrix made for the Titanic dataset using a logistic regression model.**
<img width="824" alt="Screenshot 2025-04-13 at 1 07 24 PM" src="https://github.com/user-attachments/assets/d737f069-84e7-4701-8b7a-a88491ab9c45" />

- **Below is a visualization of a ROC curve made for the Titanic dataset using a logistic regression model. A ROC curve can also be generated for the Decision Tree model.**
<img width="794" alt="Screenshot 2025-04-13 at 1 09 00 PM" src="https://github.com/user-attachments/assets/cab4e2bc-398f-4cb7-8470-8d1f8f7e4033" />



Content Suggestions:
Project Overview: Describe the goal of your project and summarize what the user can do with your app.
Instructions: Include step-by-step instructions on how to run the app locally and a link to the deployed version. List all necessary libraries and versions.
App Features: Explain the models you used and describe how hyperparameters are selected or tuned.
References: Include links to documentation, tutorials, or guides that informed your implementation.
Visual Examples: Consider adding screenshots of the app or charts produced by the app.

# Machine Learning Streamlit App 

## 📕 Project Overview 
- This interactive Streamlit app allows users to use 3 different models (decision trees, linear regression and logistic regression) to evaluate data. 3 sample datasets are loaded, but users also have the option to upload their own dataset as a CSV. Users can view various metrics including F1 scores, RMSE and R squared scores, Gini index and entropy, depending on what model is selected. 
- **Project Goal:** Demonstrate the ability to create an interactive Streamlit app that showcases the pros and cons of different machine learning models for evaluation and prediction. 
- **Skills and Packages Applied:** Python, Streamlit, Pandas, Numpy, Seaborn, Matplotlib, Scikit Learn, Graphviz, Decision Trees, Logistic Regression, and Linear Regression

## 📖 Instructions 


### 📚 Necessary Libraries and Versions

## 📲 App Features  
****The features 3 model types: Decision Trees, Logisitic Regression, and Linear Regression. A brief description of each is provided below, along with details about the hyperparameters and evaluation metrics available in the app.****
#### Decision Trees
- Can be used for classification and regression. In simple terms, a decision tree outlines options based on whether a binary variable is true or false. Each answer leads to a lower level where the process is repeated, until a conclusion (or prediction in the case of regression) can be reached, represented by a leaf node, something with no more branches
- **Hyperparameters:** The hyperparameters used in this Streamlit app are maximum depth, minimum number of samples for a split, and the app also gives the option to choose between entropy, gini index, or logloss. A brief description of what these do is provided below.
  - ***Maximum depth:*** Changes how many levels the decision tree will have, with a smaller number meaning there will be fewer levels.
  - ***Minimum samples to split:*** The minimum number of samples required to allow a split in node. This is a helpful hyperparameter to help prevent oversplitting,when the data is split too much so the leaves will not have enough samples.
  - ***Entropy vs Gini Index vs Log loss:***
#### Linear Regression 
- Can be used for linear relationships where the feature variables are numeric or categorical and the target is something on a continuous numeric scale. A linear regression model is helpful for evaluating a relationship between variables.
- 
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

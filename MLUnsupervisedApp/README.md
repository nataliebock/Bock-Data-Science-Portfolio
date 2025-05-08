App Features: Explain the models you used and describe how hyperparameters are selected or tuned.
References: Include links to documentation, tutorials, or guides that informed your implementation.
Visual Examples: Consider adding screenshots of the app or charts produced by the app.

# Unsupervised Machine Learning App 
## Project Overview
This Streamlit app encourages users to interact with different widgets in order to evaluate data with different unsupervised learning techniques (K-means Clustering, Hierarchical Clustering and Principal Component Analysis). The app features sample datasets, but also allows the user to upload their own CSV file. Users can use this app for exploratory data analysis, and can evaluate the model performances through different [metrics](#-App-Features).
- **Project goal:** The goal of this project is to demonstrate an improved use of Streamlit, while also showing an understanding of unsupervised machine learning and the pros and cons of the 3 featured models. 
- **Skills:** 
## 📖 Instructions 
This script was written in Visual Studio code. The instructions below detail how to run the code using this platform.

### 🏃 Run the App Locally: 



**1.** Import the packages and libraries detailed in the "Necessary Libraries and Versions section" below.


**2.** Run Streamlit in your terminal and open the prompted link to view the app. For more detailed instructions on this setup, click [here](https://docs.kanaries.net/topics/Streamlit/streamlit-vscode)


**3.** Once these steps are completed, you will be able to run the code in your own VS code environment! 

### 🔗 Direct Link to the Deployed Version 
To view the App without running the code, click ...ADD LINK....

### 📚 Necessary Libraries and Versions 
- **necessary libraries and packages:** Streamlit, Pandas, Numpy, Seaborn, Matplotlib, Scikit Learn, and Graphviz
 - For more specifics, view the code below:
````
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering 
````
- ***Note:*** Graphviz must also be Conda installed in Visual Studio code. For installation instructions, click [here](https://anaconda.org/conda-forge/python-graphviz)

**Versions**

  
## 📲 App Features  
**The app features 3 model types: K-means Clustering, Hierarchical Clustering, and Principal Component Analysis (PCA). A brief description of each is provided below, along with details about the hyperparameters and evaluation metrics available in the app.**

### K-means Clustering
- **Purpose:** K-means is a type of clustering that groups observations, the information provided in the rows of datasets, into clusters based on finding the best centroid positions. K-means and clustering generally is helpful for finding patterns and creating groups in the data.
- **Clusters:** Users are shown a sliding scale from 2 to 10 where they can select a number of clusters. Clusters are helpful for finding groups in the data, especially since it is unlabeled, and the model will then determine the best centroids for the selected number of groups. Based on the user's selection, a plot using PCA to reduce the number of dimensions to 2 for easy visualization is then created.
- **Evaluation:** To help the user evaluate what the optimal number of clusters is, visualizations are created showing the silhouette score and elbow method. ***Note*** while the Silhouette score is less subjective, there is no perfect metric, instead multiple metrics should be considered. Especially since unsupervised learning is part of exploratory data analysis, it is important to test different options and see which is best.
 - ***Silhouette Score:***
 - ***Elbow Method:***
 - 
### Hierarchical Clustering
- **Purpose:** Hierarchical clustering is also a type of clustering. This method is unique in that it creates a hierarchical tree of clusters, where more complicated clustering relationships can be seen. Hierarchical clustering also differs from K-means because there is no fixed k, meaning the data clusters can be of varying sizes. The tree can also be seen ***before*** determining the number of clusters, allowing for more informed analysis.
### Principal Component Analysis (PCA)
- **Purpose:** PCA is a type of dimensionality reduction unsupervised learning model. Essentially PCA and dimensionality reduction broadly **reduces the number of columns, or features,** with the goal of simplifying the data without losing too much information. If you want to simplify your dimensions, this model will be helpful.

**Scaled vs Unscaled**
- For each model option, the user is given the choice between scaling the data, or leaving the data unscaled. Using unscaled data can make it difficult to compare features measured in different units. By scaling the data, it becomes easier to draw comparisons across the different features. Scaling for all three model types is important, with the reasoning being described in depth in the app.
<img width="341" alt="Screenshot 2025-05-06 at 1 45 38 PM" src="https://github.com/user-attachments/assets/01dea17a-499b-451c-8839-9f11a22520b8" />

## ℹ️ References 
[Installing Graphviz](https://anaconda.org/conda-forge/python-graphviz)
[Streamlit App Inspiration for Machine Learning](https://varunlobo-decision-tree-using-streamlit-main-myvzpw.streamlit.app/)
[Streamlit App Format Inspiration](https://github.com/dataprofessor/dp-machinelearning/blob/master/streamlit_app.py)
[Links to different Readme sections](https://gist.github.com/rachelhyman/b1f109155c9dafffe618)
[Learn About PCA](https://www.turing.com/kb/guide-to-principal-component-analysis#what-is-pca?)
[Learn About Hierarchical Clustering](https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python)

**Learn More About the Sample Datasets**
[Penguins Sample Dataset](https://github.com/allisonhorst/palmerpenguins)
[Titanic Sample Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
[Iris Sample Dataset](https://www.kaggle.com/datasets/uciml/iris)

# Unsupervised Machine Learning App 
## Project Overview
This Streamlit app encourages users to interact with different widgets in order to evaluate data with different unsupervised learning techniques (K-means Clustering, Hierarchical Clustering and Principal Component Analysis). The app features sample datasets, but also allows the user to upload their own CSV file. Users can use this app for exploratory data analysis, and can evaluate the model performances through different [metrics](#-App-Features).
- **Project goal:** The goal of this project is to demonstrate an improved use of Streamlit, while also showing an understanding of unsupervised machine learning and the pros and cons of the 3 featured models. 
- **Skills:** Python, Streamlit, Pandas, Numpy, Seaborn, Matplotlib, Scikit Learn, Unsupervised Machine Learning, K-means, Hierarchical Clustering, and Principal Component Analysis
## 📖 Instructions 
This script was written in Visual Studio code. The instructions below detail how to run the code using this platform.

### 🏃 Run the App Locally: 



**1.** Import the packages and libraries detailed in the "Necessary Libraries and Versions section" below.


**2.** Run Streamlit in your terminal and open the prompted link to view the app. For more detailed instructions on this setup, click [here](https://docs.kanaries.net/topics/Streamlit/streamlit-vscode)


**3.** Once these steps are completed, you will be able to run the code in your own VS code environment! 

### 🔗 Direct Link to the Deployed Version 
To view the App without running the code, click [HERE](https://unsupervisedmachinelearningappnkb.streamlit.app)

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

**Versions**
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
**The app features 3 model types: K-means Clustering, Hierarchical Clustering, and Principal Component Analysis (PCA). A brief description of each is provided below, along with details about the hyperparameters and evaluation metrics available in the app.**

### K-means Clustering
- **Purpose:** K-means is a type of clustering that groups observations, the information provided in the rows of datasets, into clusters based on finding the best centroid positions. K-means and clustering generally is helpful for finding patterns and creating groups in the data.
- **Clusters:** Users are shown a sliding scale from 2 to 10 where they can select a number of clusters. Clusters are helpful for finding groups in the data, especially since it is unlabeled, and the model will then determine the best centroids for the selected number of groups. Based on the user's selection, a plot using PCA to reduce the number of dimensions to 2 for easy visualization is then created.
- **Evaluation:** To help the user evaluate what the optimal number of clusters is, visualizations are created showing the silhouette score and elbow method. ***Note*** while the Silhouette score is less subjective, there is no perfect metric, instead multiple metrics should be considered. Especially since unsupervised learning is part of exploratory data analysis, it is important to test different options and see which is best.
  - ***Silhouette Score:*** Evaluates the clustering results, with a ***higher score meaning a better clustering.*** This is accomplished by determining how well each point fits the assigned cluster in comparison to the others, and an average silhouette score of each k (number of clusters) is displayed.
  - ***Elbow Method:*** Evaluates the clustering results by plotting the within-cluster sum of squares and the number of clusters. In this case, the best number of clusters is identified by the ***elbow point***, where the decrease in WCSS flattens, or diminishes.

### Hierarchical Clustering
- **Purpose:** Hierarchical clustering is also a type of clustering. This method is unique in that it creates a hierarchical tree of clusters, where more complicated clustering relationships can be seen. Hierarchical clustering also differs from K-means because there is no fixed k, meaning the data clusters can be of varying sizes. The tree can also be seen ***before*** determining the number of clusters, allowing for more informed analysis.

**Choices for Dendrogram:** The interactive widgets created for altering the dendrogram are depicted and described below
<img width="774" alt="Screenshot 2025-05-08 at 11 41 26 AM" src="https://github.com/user-attachments/assets/b142596f-fe9c-40f7-acfd-e151bd038da2" />

  - ***Linkage***: Linkage provides different ways to measure clusters before merging. Users are given the option of selecting one of the four linkage types described below:
     - *Complete:* When comparing two clusters, the complete linkage option will look for *maximum distance between two points, one in each cluster. The drawback for complete linkage is a strong influence of outliers since it relies on maximum distances.
     - *Average:* When comparing two clusters, the average linkage option will calculate the average for every pair of points in the clusters.
     - *Single:* When comparing two clusters, the single linkage option will look for minimum distance between two points, one in each cluster. Using single linkage may create a long cluster without clearly defined groups, a drawback that may be solved by using a different type of linkage.
     - *Ward:* When comparing two clusters, the ward linkage option has the goal of minimizing the increase in the total within cluster variance created by the merging of clusters.
       
- ***Label Selection***: Users can also pick the feature that is used to label the dendrogram.
- ***Truncation***: Users can choose to truncate the dendrogram, which will group things together at a higher level and create simpler dendrogram.
- **Clusters:** Users are then given the option to select a number of clusters between 1 and 10 to create a visualization of the clusters using PCA to reduce the number of dimensions *only for visualization.*

- **Evaluation:** To help the user evaluate what the optimal number of clusters is, a visualization is created showing the silhouette score.
    - ***Silhouette Score:*** Evaluates the clustering results, with a ***higher score meaning a better clustering.*** This is accomplished by determining how well each point fits the assigned cluster in comparison to the others, and an average silhouette score of each k (number of clusters) is displayed.

### Principal Component Analysis (PCA)
- **Purpose:** PCA is a type of dimensionality reduction unsupervised learning model. Essentially PCA and dimensionality reduction broadly reduces the number of columns, or features, with the goal of simplifying the data without losing too much information. If you want to simplify the number of features in a dataset, this model will be helpful.
- **Number of Components**: Users are shown a sliding scale from 2 to 10 where they can select a number of components they would like to use.
- **Evaluation**: To evaluate the PCA, users are shown an explained variance plot that features individual and cumulative variance.
   - ***Cumulative***: The cumulative variance explains uses the elbow method to determine the ideal number of components. The elbow method evaluates by plotting the within-cluster sum of squares. In this case, the best number of clusters is identified by the elbow point, where the decrease in WCSS flattens, or diminishes.
The bars below shows the individual variance explained, showing the variance that each separate component is able to explain. This is helpful for determining which components to use for further analysis.
   - ***Individual***: The individual variance explained shows the variance that each separate component is able to explain. This is helpful for determining which components to use for further analysis.
### Scaled vs Unscaled
- For each model option, the user is given the choice between scaling the data, or leaving the data unscaled. Using unscaled data can make it difficult to compare features measured in different units. By scaling the data, it becomes easier to draw comparisons across the different features. Scaling for all three model types is important, with the reasoning being described in depth in the app. While scaling is strongly encouraged for each model, offering the choice can help users see *why* scaling is so important in most cases, and make the app work in a case where unscaled data is necessary. 
  
<img width="341" alt="Screenshot 2025-05-06 at 1 45 38 PM" src="https://github.com/user-attachments/assets/01dea17a-499b-451c-8839-9f11a22520b8" />

## ℹ️ References 
[Installing Graphviz](https://anaconda.org/conda-forge/python-graphviz)
[Streamlit App Inspiration for Machine Learning](https://varunlobo-decision-tree-using-streamlit-main-myvzpw.streamlit.app/)
[Streamlit App Format Inspiration](https://github.com/dataprofessor/dp-machinelearning/blob/master/streamlit_app.py)
[Links to different Readme sections](https://gist.github.com/rachelhyman/b1f109155c9dafffe618)
[Learn About PCA](https://www.turing.com/kb/guide-to-principal-component-analysis#what-is-pca?)
[Learn About Hierarchical Clustering](https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python)
[Hierarchical Clustering, especially linkage](https://r.qcbs.ca/workshop09/book-en/clustering.html)
[Grokking Machine Learning Chapter 2 for Unsupervised](https://www.manning.com/books/grokking-machine-learning)


**Learn More About the Sample Datasets**
[Penguins Sample Dataset](https://github.com/allisonhorst/palmerpenguins)
[Titanic Sample Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
[Iris Sample Dataset](https://www.kaggle.com/datasets/uciml/iris)
[Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

## Visual Examples 
- **An example of a Silhouette Score graph made using K-means clustering**
<img width="745" alt="Screenshot 2025-05-08 at 10 52 21 AM" src="https://github.com/user-attachments/assets/dccdd554-232f-40d7-9ad7-dae07538c62e" />

- **An example of a of the Hierarchical Clustering visualization using PCA for dimensionality reduction for plotting in 2D. In this example, the user selected 3 clusters**
 <img width="748" alt="Screenshot 2025-05-08 at 12 06 29 PM" src="https://github.com/user-attachments/assets/ff1f6e20-d6bf-43e3-8d27-afed8683d436" />

- **An example of a Dendrogram made using Hierarchical clustering. Here, the user selected the complete linkage option and chose to truncate.**
<img width="760" alt="Screenshot 2025-05-08 at 12 02 04 PM" src="https://github.com/user-attachments/assets/3a976c3a-8a33-45f6-bb0e-8ea581eeb8bd" />

- **An example of an explained variance visualization for PCA where the user selected 5 components**
<img width="756" alt="Screenshot 2025-05-08 at 12 56 28 PM" src="https://github.com/user-attachments/assets/ee7bcefb-5422-4ffa-be47-74645fb19582" />


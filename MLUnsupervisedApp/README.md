Content Suggestions:
Project Overview: Describe the goal of your project and summarize what the user can do with your app.
Instructions: Include step-by-step instructions on how to run the app locally and a link to the deployed version. List all necessary libraries and versions.
App Features: Explain the models you used and describe how hyperparameters are selected or tuned.
References: Include links to documentation, tutorials, or guides that informed your implementation.
Visual Examples: Consider adding screenshots of the app or charts produced by the app.

# Unsupervised Machine Learning App 
## Project Overview

## 📖 Instructions 
This script was written in Visual Studio code. The instructions below detail how to run the code using this platform.

### 🏃 Run the App Locally: 



**1.** Import the packages and libraries detailed in the "Necessary Libraries and Versions section" below.


**2.** Run Streamlit in your terminal and open the prompted link to view the app. For more detailed instructions on this setup, click [here](https://docs.kanaries.net/topics/Streamlit/streamlit-vscode)


**3.** Once these steps are completed, you will be able to run the code in your own VS code environment! 

### 🔗 Direct Link to the Deployed Version 
To view the App without running the code, click ...ADD LINK....

## 📲 App Features  
**The app features 3 model types: K-means Clustering, Hierarchical Clustering, and Principal Component Analysis (PCA). A brief description of each is provided below, along with details about the hyperparameters and evaluation metrics available in the app.**
**Scaled vs Unscaled**
- For each model option, the user is given the choice between scaling the data, or leaving the data unscaled. Using unscaled data can make it difficult to compare features measured in different units. By scaling the data, it becomes easier to draw comparisons across the different features. Scaling for all three model types is important, with the reasoning being described in depth in the app.
<img width="341" alt="Screenshot 2025-05-06 at 1 45 38 PM" src="https://github.com/user-attachments/assets/01dea17a-499b-451c-8839-9f11a22520b8" />

### K-means Clustering
- **Purpose:** K-means is a type of clustering that groups observations, the information provided in the rows of datasets, into clusters based on finding the best centroid positions. K-means and clustering generally is helpful for finding patterns and creating groups in the data.
### Hierarchical Clustering
- **Purpose:** Hierarchical clustering is also a type of clustering. This method is unique in that it creates a hierarchical tree of clusters, where more complicated clustering relationships can be seen. Hierarchical clustering also differs from K-means because there is no fixed k, meaning the data clusters can be of varying sizes. The tree can also be seen ***before*** determining the number of clusters, allowing for more informed analysis.
### Principal Component Analysis (PCA)
- **Purpose:** PCA is a type of dimensionality reduction unsupervised learning model. Essentially PCA and dimensionality reduction broadly **reduces the number of columns, or features,** with the goal of simplifying the data without losing too much information. If you want to simplify your dimensions, this model will be helpful.


## ℹ️ References 
[Installing Graphviz](https://anaconda.org/conda-forge/python-graphviz)
[Streamlit App Inspiration for Machine Learning](https://varunlobo-decision-tree-using-streamlit-main-myvzpw.streamlit.app/)
[Streamlit App Format Inspiration](https://github.com/dataprofessor/dp-machinelearning/blob/master/streamlit_app.py)
[Links to different Readme sections](https://gist.github.com/rachelhyman/b1f109155c9dafffe618)
[Learn About PCA](https://www.turing.com/kb/guide-to-principal-component-analysis#what-is-pca?)



**Learn More About the Sample Datasets**
[Penguins Sample Dataset](https://github.com/allisonhorst/palmerpenguins)
[Titanic Sample Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
[Iris Sample Dataset](https://www.kaggle.com/datasets/uciml/iris)

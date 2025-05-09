##Unsupervised Machine Learning Streamlit App

#Importing necessary Packages
##For me, it is helpful to do most separately so I can easily keep track of which packages and libraries are loaded. 
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
# Creating a title for app and adding an explanation 
st.title("Unsupervised Machine Learning App")
st.subheader("Welcome to this Unsupervised Machine Learning App!")
st.markdown("**Getting started:** Select a sample dataset or upload your own CSV on the left side panel. You will then be shown a preview of the data, and can begin making choices about what type of model you would like to use. Some metrics for evaluation are shown, as well as some brief explanations of what different metrics show. For a more in depth understanding, you can view the reading materials linked on the connected Github README file.")
st.markdown("**What is Unusupervised Machine Learning?** Machine Unsupervised learning uses unlabeled data consisting only of features. It is used for purposes including exploratory data analysis and preprocessing data. Two categories of unsupervised learning will be presented here: dimensionality reduction (PCA) and clustering (hierarchical and k-means).")


##Data
#Uploading user dataset code
st.sidebar.header("Upload or Choose a Dataset") #adding it to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) #upload file as csv
sam_data = st.sidebar.selectbox("Or choose a sample dataset", ["None", "Breast Cancer", "Iris", "Titanic", "Penguins"]) #sample dataset options and default as none

# Loading sample datasets and creating the options for datasets. First, if user uploads dataset, then breast cancer dataset, Iris dataset, Titanic dataset, and finally penguins
df = None
if uploaded_file: 
    df = pd.read_csv(uploaded_file) #If own dataset uploaded load this
    df = pd.get_dummies(df, columns= df.select_dtypes(exclude=['number']).columns.tolist(), drop_first=True) #changing categorical columns into dummy, pd.get_dummies converts categorical variables into dummy variables so have numeric variables, use drop_first to avoid dummy trap. Using exclude=['number'] so only applies to non-numeric columns and makes them into dummies. 
    df= df.fillna(df.median())#dropping the na values for uploaded datasets dropped too many when just dropping all rows with any missing data. So I decided to impute the median value instead 
    dfname = "Your Dataset"
elif sam_data == "Breast Cancer":
    breast_cancer = load_breast_cancer() 
    X = breast_cancer.data #making sure this data is loaded in properly as a dataset by setting X as a feature matrix
    y = breast_cancer.target #while NO target is used, decided to use this lingo for setting up the dataframe 
    feature_names = breast_cancer.feature_names
    target_names = breast_cancer.target_names
    df = pd.DataFrame(X, columns=feature_names) #making dataframe 
    df['target'] = y  #creating a 'target' column in the dataset
    dfname = "Breast Cancer" #if Breask cancer selected load this
    st.sidebar.markdown("**About Breast Cancer Dataset:** Contains information about Breast Cancer in Wisconsin, specifically the characteristics of tumors.") #adding brief info about data
    st.sidebar.markdown("For more information click [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)") #adding hyperlink to more information about dataset
elif sam_data == "Iris":
    df = sns.load_dataset("iris") #if Iris selected load this
    df = df[df["species"].isin(["setosa", "virginica"])] #changing the species to only have two options so it works with binary classification
    df = pd.get_dummies(df, columns= df.select_dtypes(exclude=['number']).columns.tolist(), drop_first=True) #same as with uploaded_file
    dfname = "Iris"
    st.sidebar.markdown("**About Iris Dataset:** Contains information about Iris flowers, specifically their measurements.") #adding brief info about data
    st.sidebar.markdown("For more information click [here](https://www.kaggle.com/datasets/uciml/iris)") #adding hyperlink to more information about dataset
elif sam_data == "Titanic":
    df = sns.load_dataset("titanic").dropna(axis=1)  #if titanic selected load this 
    df = pd.get_dummies(df, columns= df.select_dtypes(exclude=['number']).columns.tolist(), drop_first=True)
    dfname = "Titanic"
    st.sidebar.markdown("**About Titanic Dataset:** Contains information about passengers on the Titanic, including their class, fare price, gender and whether they survived.") #including brief info about the data 
    st.sidebar.markdown("For more information click [here](https://www.kaggle.com/datasets/yasserh/titanic-dataset)")
elif sam_data == "Penguins":
    df = sns.load_dataset("penguins") 
    df = df[df["species"].isin(["Gentoo", "Chinstrap"])] #changing the species to only have two options so it works with binary classification
    df.dropna(inplace=True)
    dfname = "Penguins"
    df = pd.get_dummies(df, columns= df.select_dtypes(exclude=['number']).columns.tolist(), drop_first=True)
    st.sidebar.markdown("**About Penguins Dataset:** Contains information about Penguin species living on different islands.")
    st.sidebar.markdown("For more information click [here](https://github.com/allisonhorst/palmerpenguins)")

if df is not None:
    st.write("**Preview of Dataset**", df.head()) #show a preview of df selected 

    #Selecting features here, also giving option to select a "target column"/ label column if want to compare the clusters with label column
    sel_col = None #since unsupervised learning as no target column since unlabeled, setting this to None 
    features = st.multiselect("**Select the features you would like to use:**", options=[col for col in df.columns if col != sel_col]) #allowing users to select multiple feature columns, besides the sel_col / the target column previously selected 
    if features:
        X = df[features] #defining X 
        y = None #since no target column for unsupervised learning

    #Model options 
    #providing brief descriptions of the 3 models offered
        st.header("Selecting a Model")
        st.markdown("**Choose which unsupervised machine learning model you would like to use. Below is a brief explanation of each model:**")
        st.markdown("- **K-means clustering:** K-means is a type of clustering that groups observations, the information provided in the rows of datasets, into clusters based on finding the best centroid positions. K-means and clustering generally is helpful for finding patterns and creating groups in the data.")
        st.markdown("- **Hierarchical Clustering:** Hierarchical clustering is also a type of clustering. This method is unique in that it creates a hierarchical tree of clusters, where more complicated clustering relationships can be seen. Hierarchical clustering also differs from K-means because there is no fixed k, meaning the data clusters can be of varying sizes. The tree can also be seen ***before*** determining the number of clusters, allowing for more informed analysis. ")
        st.markdown("- **Principal Component Analysis:** PCA is a type of dimensionality reduction unsupervised learning model. Essentially PCA and dimensionality reduction broadly reduces the number of columns, or features, with the goal of simplifying the data without losing too much information. If you want to simplify your dimensions, this model will be helpful.")
        mlselect = st.selectbox("", ["k-means clustering", "hierarchical clustering", "principal component analysis"]) #creating a selectbox so users can pick which model they want to use



##k-means clustering
#throughout the creation of the models, if, else and else if "elif" statements are used to make the code conditional on other parts. This is helpful since there are three model options to pick from 
        if mlselect == "k-means clustering": #if k-means clustering selected, this code will be run
            st.markdown("You've selected K-means, a type of clustering unsupervised learning. Below, you can decide whether to scale the data and adjust the number of clusters.") #text that will appear k-means clustering is selected as the model. 
            st.subheader("Scaling") #giving users the the option to scale data and providing explanations for what scaling does
            st.markdown("- Using unscaled data can make it difficult to compare features measured in different units. By scaling the data, it becomes easier to draw comparisons across the different features.")
            st.markdown("- If the features are already in the same units, scaling may be unnecessary. It is important to consider what is necessary for your data, and your objectives.")
            st.markdown("**NOTE:** For K-means clustering, distance calculations are used and can be influenced by the scale of features.")
            app_scale = st.radio("**Select whether you would like to scale the model**", ["Unscaled", "Scaled"]) #giving users the option to pick between scaled and unscaled through radio prompt with a circle for each option
            if app_scale == "Scaled": #if scaled, apply scaler 
                scaler = StandardScaler() #using standardscaler    
                X_std = scaler.fit_transform(X) ##since the clustering is based on how close they are to centroid, need to make sure to scale so all in same unit. Standard scaling looks at means across, and then does standard deviation by comparing against the mean 
            else:
                X_std = X.values #if unscaled, will define X_std as the unscaled x values so still works with rest of code 
            st.subheader("Clustering")
            st.markdown("- **Clusters:** Clusters are helpful for finding groups in the data, especially since it is unlabeled. The model will determine the best centroids for the determined number of groups.") #explaining why clusters / what they do
            k = st.slider('Number of clusters (k)', 1, 10, 3) #set k to some number of clusters chosen on slider
            kmeans = KMeans(n_clusters = k) #number of clusters = whatever number chosen on the slider 
            kmeans.fit_predict(X_std) #finds the centroids and then clusters
            clusters = kmeans.fit_predict(X_std) #defining clusters with kmeans
#visualizing the model
            st.subheader("Visualizing the Results:") 
            st.markdown("- The scatterplot below shows how the model has clustered the data. Since the data has a large number of dimensions, PCA is used in order to reduce to 2D and allow for plotting.")
            st.markdown("- ***Reminder:*** PCA, prinipal component analysis, is a dimension reduction technique so it simplifies the features, in this case down to 2 to allow for visualization.")
            st.markdown("- The axes show the two dimensions, with the x-axis representing the first component and the y-axis showing the second component.")
            if X_std.shape[1] < 2: #making sure that the user has selected enough features to do pca, so program an error if less than 2 selected
                st.error("Please select a minimum of 2 features above to perform PCA.") #creating an error message to ensure that users select more than 2 features for PCA to graph
            else:
                pca = PCA(n_components=2)# = 2 so can visualize in 2D, easier to see.
                X_pca = pca.fit_transform(X_std) #fitting X_PCA 

            # Create a scatter plot of the PCA-transformed data, colored by KMeans cluster labels
            plt.figure(figsize=(8, 6)) #setting the size of the figure
            color_choices = ['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'purple', 'black', 'cyan', 'magenta'] #providing color options so that no matter how many clusters the user selects, there is a color. Since allow to pick up to 10 clusters, there are 10 colors
            for k in np.unique(clusters): 
                plt.scatter(X_pca[clusters == k, 0], X_pca[clusters == k, 1], #creating the scaterplot for the clustes
                    c=color_choices[k % len(color_choices)], alpha=0.7, edgecolor='k', s=60, label=f'Cluster {k}') #applying the color choices, and using alpha to make points slightly transparent
            plt.xlabel('Principal Component 1') #labeling the x axis
            plt.ylabel('Principal Component 2') #labeling the y axis 
            plt.title('KMeans Clustering: 2D PCA Projection') #title of the graphic 
            plt.legend(loc='best') #setting the legend so that matlab picks where is best to place based on cluster locations 
            plt.grid(True)
            st.pyplot(plt) #so that streamlit plots the graph

#elbow vs silhouette 
            st.subheader("Finding the Optimal Number of Clusters:") 
            st.markdown("The two visualization options below can help determine the optimal number of clusters for the selected dataset. Select an option to view.")
            st.markdown("- **Silhouette Score:** Evaluates the clustering results, with a ***higher score meaning a better clustering.*** This is accomplished by determining how well each point fits the assigned cluster in comparison to the others, and an average silhouette score of each k (number of clusters) is displayed.")
            st.markdown("- **Elbow Method:** Also evaluates the clustering results by plotting the within-cluster sum of squares and the number of clusters. In this case, the best number of clusters is identified by the ***elbow point***, where the decrease in WCSS flattens, or diminishes.")
            st.markdown("**NOTE:** While the Silhouette score is less subjective, there is no perfect metric, instead multiple metrics should be considered. Especially since unsupervised learning is part of exploratory data analysis, it is important to test different options and see which is best.")
            el_sil = st.radio("**Select An Evaluation Method for Determining the Optimal Number of Clusters**", ["Silhouette Score", "Elbow Method"]) #giving users the option to pick between two methods for evaluation
            if el_sil == "Silhouette Score": #if silhouette, 
                ks = range(2,11) #setting the range of K's shown on the graph. Since the options are 2-10, showing 2-10
                wcss = [] #for within cluster sum of squares for elbow method
                silhouette_scores = []
                for k in ks: # Silhouette scores for each k
                    km = KMeans(k) #fit kmeans model, will loop through doing k = 2,3,4,5,etc
                    km.fit(X_std) #just fitting the model ,then can take inertia attribute
                    wcss.append(km.inertia_) #inertia: sum of squared distances within clusters
                    labels = km.labels_ #grab the labels
                    silhouette_scores.append(silhouette_score(X_std, labels)) #looking at outside of cluster score for each point and labels, will grab scores
#plotting the sil score
                plt.figure(figsize=(8,6)) #setting the size of the figure
                plt.plot(ks, silhouette_scores, marker='o', color='maroon') #setting the color and creating the plot
                plt.xlabel('Number of clusters (k)') #x axis label
                plt.ylabel('Silhouette Score') #y axis label
                plt.title('Silhouette Score for Optimal k') #plot title
                plt.grid(True) #creating a grid 
                plt.tight_layout() #automatic adjustments to fit in the figure area 
                st.pyplot(plt) #showing the plot in streamlit
#creating the elbow method plot 
            else: 
                ks = range(2,11) #looking at k = all these numbers since the options are 2-10, showing 2-10
                wcss = [] #for wcss
                for k in ks:
                    km = KMeans(k) #defining kmeans 
                    km.fit(X_std) #fitting kmeans
                    wcss.append(km.inertia_)  #inertia: sum of squared distances within clusters
                    labels = km.labels_ #adding labels from kmeans 
                plt.figure(figsize=(8,6)) #setting the size of the figure, same as Sil Score so easy to compare
                plt.plot(ks, wcss, marker='o', color = 'navy') #setting the color and running the plot
                plt.xlabel('Number of clusters (k)') #xaxis
                plt.ylabel('Within-Cluster Sum of Squares (WCSS)') #yaxis
                plt.title('Elbow Method for Optimal k') #title 
                plt.grid(True) #showing grid
                st.pyplot(plt) #plotting it in streamlit 

##Principal Component Reduction
        elif mlselect == "principal component analysis": #if principal component analysis selected 
            st.markdown("You've selected principal component analysis, below you can adjust the number of components, decide whether to scale the data, and view the explained variance.") #what will appear when selecting pca
            st.subheader("Components") #explaining what components do for PCA 
            st.markdown("Below you can select the number of components to use in the PCA, having a smaller number of components means that some information will be lost. **The number of components cannot be greater than the number of features (columns) chosen for use in the analysis.**")
            com_choice = st.slider('**Number of Components**', 1, 10, 3) #allowing user to pick number of components 
            st.subheader("Scaling") #giving users the the option to scale data and providing explanations for what scaling does
            st.markdown("- Using unscaled data can make it difficult to compare features measured in different units. By scaling the data, it becomes easier to draw comparisons across the different features. For PCA, scaling is also important because it ensures that features on a larger numeric scale do not skew the impact and variance.") #explaining scaled vs unscaled data
            st.markdown("- If the features are already in the same units, scaling may be unnecessary. It is important to consider what is necessary for your data, and your objectives.")
            st.markdown("**NOTE:** PCA is sensitive to the scale of features, it is ***highly recommended*** to scale the data.") #encouraging scaling! 
            app_scale = st.radio("**Select whether you would like to scale the model**", ["Unscaled", "Scaled"]) #giving users the option to pick between scaled and unscaled through radio prompt with a circle for each option
            if app_scale == "Scaled": #if scaled, apply scaler 
                scaler = StandardScaler() #using standardscaler    
                X_std = scaler.fit_transform(X) ##since the clustering is based on how close they are to centroid, need to make sure to scale so all in same unit. Standard scaling looks at means across, and then does standard deviation by comparing against the mean 
            else:
                X_std = X.values #if unscaled, will define X_std as the unscaled x values so still works with rest of code 
            #kmeans.fit_predict(X_std) #finds the centroids and then clusters
            #clusters = kmeans.fit_predict(X_std)
            pca = PCA(n_components= com_choice) #having the number of components for PCA reflect user choice from the slider defined as com_choice 
            X_pca = pca.fit_transform(X_std) #fitting the model 
#Explained variance combined plot
            st.subheader("Explained Variance") #explaining what the plot shows and what explained variance is 
            st.markdown("- The explained variance plot below shows a visualization of the cumulative variance to help the user determine the best number of components for the selected data set, a helpful insight for knowing how many components to use in future analysis.")
            st.markdown("- **The green line** in the graph below uses the elbow method to determine the ideal number of components as ***cumulative variance*** explained. The elbow method evaluates by plotting the within-cluster sum of squares. In this case, the best number of clusters is identified by the ***elbow point***, where the decrease in WCSS flattens, or diminishes.")
            st.markdown("- **The bars** below shows the ***individual variance*** explained, showing the variance that each separate component is able to explain. This is helpful for determining which components to use for further analysis.")
            fig, ax1 = plt.subplots(figsize=(8, 6)) #setting the figure size 
            pca_sc = PCA(n_components=com_choice).fit(X_std) #making it so the number of components selected by user is the number reflected in PCA visual
            cumulative_variance = np.cumsum(pca_sc.explained_variance_ratio_) #calculating cumulative variance
            explained = pca_sc.explained_variance_ratio_ * 100  # variance for each component as a percentage 
            components = np.arange(1, len(explained) + 1) #starting number of components at 1 and ending at length chosen in explained percentage 
            cumulative = np.cumsum(explained) #the cumulative sum for the number defined in explained
            bar_color = 'purple' #setting the color
            ax1.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')  #creating the bars for individual variance 
            ax1.set_xlabel('Principal Component') #labeling the x-axis underneath the bars 
            ax1.set_ylabel('Individual Variance Explained (%)', color=bar_color) #labeling the individual variance bar
            ax1.tick_params(axis='y', labelcolor=bar_color) 
            ax1.set_xticks(components)
            for i, v in enumerate(explained):
                ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black') #adding percentages to each bar
            ax2 = ax1.twinx()
            line_color = 'green' #defining line color 
            ax2.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance') #second y axis for cumulative variance 
            ax2.set_ylabel('Cumulative Variance Explained (%)', color=line_color) #setting y axis 
            ax2.tick_params(axis='y', labelcolor=line_color) #setting the axis and color of the axis text
            ax2.set_ylim(0, 100) #limiting y to 100 since shown as percentages out of 100 
            ax1.grid(False) #removing grid lines
            ax2.grid(False) #removing grid lines
            lines1, labels1 = ax1.get_legend_handles_labels() # Combined legends
            lines2, labels2 = ax2.get_legend_handles_labels()# Combined legends
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best') #setting the legend to best location found, 
            plt.title('PCA: Variance Explained') #title for graph
            plt.tight_layout() #automatic adjustments to fit in the figure area 
            st.pyplot(plt) #showing the plot 
# Show Explained Variance Ratios numerically 
            st.markdown("The boxes below show the exact numeric values of the explained variance and cumulative explained variance plotted in the visualization above.") #explaining the numbers shown below
            explained_variance = pca.explained_variance_ratio_ #calculating explained variance 
            st.write("Explained Variance Ratio:", explained_variance) #printing the explained variance 
            st.write("Cumulative Explained Variance:", np.cumsum(explained_variance)) #printing the cumulative explained variance 

##Hierarchical clustering
        elif mlselect == "hierarchical clustering":
            st.markdown("You've selected hierarchical clustering, a type of clustering unsupervised machine learning. Below, you can decide whether to scale you data and also interact with different widgets to create a dendrogram.") #what will appear when selecting HC
            st.subheader("Scaling") #giving users the the option to scale data and providing explanations for what scaling does
            st.markdown("- Using unscaled data can make it difficult to compare features measured in different units.")
            st.markdown("- If the features are already in the same units, scaling may be unnecessary. It is important to consider what is necessary for your data, and your objectives.")
            st.markdown("**NOTE:** Hierarchical clustering uses Euclidean distance to find the shortest distance between points. Therefore, using a scaler is important because it ensures that features on a larger scale do not have a disproportionate impact.")
            app_scale = st.radio("**Select whether you would like to scale the model**", ["Unscaled", "Scaled"]) #giving users the option to pick between scaled and unscaled through radio prompt with a circle for each option
            if app_scale == "Scaled": #if scaled, apply scaler 
                scaler = StandardScaler() #using standardscaler    
                X_std = scaler.fit_transform(X) ##since the clustering is based on how close they are to centroid, need to make sure to scale so all in same unit. Standard scaling looks at means across, and then does standard deviation by comparing against the mean 
            else:
                X_std = X.values #if unscaled, will define X_std as the unscaled x values so still works with rest of code 
        ##initial dendrogram
        #dendrogram widgets 
            st.subheader("Creating the Dendrogram")
            st.markdown("The dendrogram below helps to show how the data has been clustered. This visual can be helpful for determining what the ideal number of clusters is for the data.") #adding a brief explanation of what a dendrogram is 
            st.markdown("Below you can interact with 3 different widgets to determine what kind of linkage you would like to use, the feature you want to label the dendrogram, and whether or not you would like to truncate the dendrogram.") #general explanation of what can be done with the widgets
            st.markdown("**Linkage:** The different types of linkage offered below are different ways to measure clusters before merging:")
            st.markdown("- **Complete:** When comparing two clusters, the complete linkage option will look for ***maximum distance** between two points, one in each cluster. The drawback for complete linkage is a strong influence of outliers since it relies on maximum distances.")
            st.markdown("- **Average:** When comparing two clusters, the average linkage option will calculate the ***average*** for every pair of points in the clusters.")
            st.markdown("- **Single:** When comparing two clusters, the single linkage option will look for ***minimum distance*** between two points, one in each cluster. Using single linkage may create a long cluster without clearly defined groups, a drawback that may be solved by using a different type of linkage.")
            st.markdown("- **Ward:** When comparing two clusters, the ward linkage option  has the goal of minimizing the increase in the total within cluster variance created by the merging of clusters.")
            st.markdown("**To learn more about linkage options in hierarchical clustering, click [here](https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python) and [here](https://r.qcbs.ca/workshop09/book-en/clustering.html)**")
            link_opt = st.radio("**Select a linkage type:**", ["complete", "average", "single", "ward"]) #creating an option where users can select different linkage types 
            if link_opt == "complete": 
                linkage(X_std, method= "complete") #so if the user selects complete, Z will use complete method 
            elif link_opt == "average":
                Z = linkage(X_std, method= "average") #if user selects average, Z will be the average method
            elif link_opt == "single":
                Z = linkage(X_std, method="single") #if user selects single, Z will use the single method 
            elif link_opt == "ward":
                Z = linkage(X_std, method = "ward") #if user selects Ward, linkage matrix will use ward

            link_all = linkage(X_std, method = link_opt) #when making linkage matrix, standard to use Z. Setting method = to the linkage option chosen above so reflects user input
            st.markdown("**Label Selection:** The feature you will be used to label your dendrogram, try different options and see how the bottom labels changes")
            label_choice = st.selectbox("**Click on the feature/column you would like to use to label your dendrogram**", df.columns)
            label_choice_clean = df[label_choice].astype(str).values #converting to strings so works well with numbered features as well 
            st.markdown("**Truncate:** Truncation will group things together at a higher level and create simpler dendrogram")
            trunc_opt = st.radio("**Truncate dendrogram**", ["Yes", "No"], index=0) #giving users the option to select whether to truncate the dendrogram 
            trun_yes = "lastp" if trunc_opt == "Yes" else None #making application of lastp condition on the user's answer to trunc_opt so if yes will apply, but if not, will not apply
            plt.figure(figsize = (15,5)) #making figure size larger so able to show more of the dendrogram, usually very complex even after truncate 
            dendrogram(link_all, labels = label_choice_clean,
                       truncate_mode = trun_yes) #creating the dendrogram and making sure that user input translates by setting equal to the previous options for choice 
            st.pyplot(plt) #showing the dendrogram 
            st.subheader("Visualizing The Clusters") #clusters section 
            st.markdown("Now that you have visualized the initial dendrogram, select the number for k you would like to use")
            k = st.slider('**Number of clusters**', 1, 10, 3) #set k to some number of clusters chosen on slider and allowing users to select their own k
            agg = AgglomerativeClustering(n_clusters = k, linkage = link_opt) #creating agg clustering by the number of clusters selected, and the linkage option chosen previously
            df["Cluster"] = agg.fit_predict(X_std) #will create cluster labels for us and putting it in a new column
            #df["Cluster"].value_counts() #each cluster associated with country
            cluster_labels = df["Cluster"] #create labels based on the clusters 
            st.markdown("- Below, PCA is used to visualize the data. Since the data has a large number of dimensions, PCA is used in order to reduce to 2D and allow for plotting. *PCA was not used for fitting the clusters*.")
            st.markdown("- ***Reminder:*** PCA, prinipal component analysis, is a dimension reduction technique so it simplifies the features, in this case down to 2 to allow for visualization.")
            st.markdown("- The axes show the two dimensions, with the x-axis representing the first component and the y-axis showing the second component.")
            pca = PCA(n_components=2) #creating the pca for making the graph only so setting n_compnents to 2 so 2 dimensional and easy to visualize 
            X_pca = pca.fit_transform(X_std) #fitting the model 
            plt.figure(figsize=(10, 7)) #making the figure size 
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=60, edgecolor='k', alpha=0.7, cmap = 'rainbow') #Using cmap to add a color scheme or rainbow for the created clusters
            plt.xlabel('Principal Component 1') #x axis label
            plt.ylabel('Principal Component 2') #y axis label 
            plt.title(f'Agglomerative Clustering on (via PCA)') #creating a title that will work with any dataset by putting the df title in 
            plt.legend(*scatter.legend_elements(), title="Clusters") #creating the legend and legend title 
            plt.grid(True)
            st.pyplot(plt) #plotting 

#sil score for hierarchical clustering
            st.subheader("Determining the Optimal Number of K") #section explaining what sil score is 
            st.markdown("The visualization below can help determine the optimal number of clusters for the selected dataset.")
            st.markdown("- **Silhouette Score:** Evaluates the clustering results, with a ***higher score meaning a better clustering.*** This is accomplished by determining how well each point fits the assigned cluster in comparison to the others, and an average silhouette score of each k (number of clusters) is displayed.")
            ks = range(2,11) #looking at k = all these numbers in contained in the slider for K selection  2 - 10
            silhouette_scores = [] 
            for k in ks: # Silhouette scores for ks 
                labels = AgglomerativeClustering(n_clusters=k, linkage= link_opt).fit_predict(X_std) #using the same fit as with the dendrogram created 
                silhouette_scores.append(silhouette_score(X_std, labels)) #looking at outside of cluster score for each point and labels, will grab scores
            #plotting
            plt.figure(figsize=(8,6))
            plt.plot(ks, silhouette_scores, marker='o', color='purple') #selecting colors, applying ks from before 
            plt.xlabel('Number of Clusters (k)') #x axis label
            plt.ylabel('Average Silhouette Score') #y axis label 
            plt.title("Silhouette Analysis using Agglomerative Clustering") #graph title 
            plt.grid(True)
            st.pyplot(plt) #plotting 
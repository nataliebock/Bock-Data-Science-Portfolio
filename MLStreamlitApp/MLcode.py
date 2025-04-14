##Importing necessary Packages
##For me, it is helpful to do most separately so I can easily keep track of which packages and libraries are loaded. 
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
# Creating a title
st.title("Supervised Machine Learning App")
st.markdown("Welcome to this Machine Learning App! To get started, select a sample dataset or upload your own CSV on the left side panel. You will then be shown a preview of the data, and can begin making choices about what type of model you would like to use. Some metrics for evaluation are shown, as well as some brief explanations of what different metrics show. For a more in depth understanding, you can view the reading materials linked on the connected Github README file.")

##Data
#Uploading own dataset code
st.sidebar.header("Upload or Choose a Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sam_data = st.sidebar.selectbox("Or choose a sample dataset", ["None", "Iris", "Titanic", "Penguins"])

# Loading sample datasets and creating the options for datasets. First, if user uploads dataset, second is Iris dataset, third is Titanic dataset 
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file) #If own dataset uploaded load this
elif sam_data == "Iris":
    df = sns.load_dataset("iris") #if Iris selected load this
    df = df[df["species"].isin(["setosa", "virginica"])] #changing the species to only have two options so it works with binary classification
    st.sidebar.markdown("**About Iris Dataset:** Contains information about Iris flowers, specifically their measurements. ***If using the logistic regression or decision tree model, it is recommended that you select species as the target column.***")
elif sam_data == "Titanic":
    df = sns.load_dataset("titanic").dropna(axis=1)  #if titanic selected load this 
    st.sidebar.markdown("**About Titanic Dataset:** Contains information about passengers on the Titanic, including their class, fare price, gender and whether they survived. ***If using the logistic regression or decision tree model, it is recommended that you select survived as the target column.***")
elif sam_data == "Penguins":
    df = sns.load_dataset("penguins")
    df = df[df["species"].isin(["Gentoo", "Chinstrap"])] #changing the species to only have two options so it works with binary classification
    df.dropna(inplace=True)
    st.sidebar.markdown("**About Penguins Dataset:** Contains information about Penguin species living on different islands. ***If using the logistic regression or decision tree model, it is recommended that you select species OR sex as the target column.***")

if df is not None:
    st.write("**Preview of Dataset**", df.head()) #show a preview of df selected 

    # Selecting columns
    sel_col = st.selectbox("Select the target column", df.columns) #if only want to use some columns, selecting here
    features = st.multiselect("Select the features you would like to use:", options=[col for col in df.columns if col != sel_col])

    if features:
        X = df[features]
        y = df[sel_col]

        #Splitting the data into train and test groups 
        split = st.slider("Split the Data into Training and Testing", 10, 100, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

        #Model options 
        #providing brief descriptions of the 3 models offered
        st.header("Selecting a Model")
        st.markdown("**Choose which machine learning model you would like to use. Below is a brief explanation of each model:**")
        st.markdown("- **Decision Tree:** Can be used for classification and regression. In simple terms, a decision tree outlines options based on whether a binary variable is true or false. Each answer leads to a lower level where the process is repeated, until a conclusion (or prediction in the case of regression) can be reached, represented by a leaf node, something with no more branches. ***Select a categorical target variable***")
        st.markdown("- **Linear Regression:** Can be used for linear relationships where the feature variables are numeric or categorical and the target is something on a continuous numeric scale. A linear regression model is helpful for evaluating a relationship between variables. ***Select a target variable that is on a continuous numeric scale***")
        st.markdown("- **Logistic Regression:** Can be used with binary variables to evaluate how these features influence the probability of something happening. A logistic regression may be the best option if the goal is to see how each selected feature impact the outcome's probability. ***Select a binary target column***")
        mlselect = st.selectbox("", ["Decision Tree", "Linear Regression", "Logistic Regression"])
##Decision tree
#throughout the creation of the models, if, else and else if "elif" statements are used to make the code conditional on other parts. This is helpful since there are three model options to pick from 
        if mlselect == "Decision Tree": #if decision tree selected, this code will be run
            st.markdown("You've selected a decision tree. Below, you can adjust the max depth and criteria.")
            st.subheader("Hyperparameters")
            st.markdown("- **Maximum depth:** Changes how many levels the decision tree will have, with a smaller number meaning there will be fewer levels. ***Tip: It may be a good idea to set max depth to number of features involved***")
            st.markdown("- **Minimum Samples to Split:** The minimum number of samples required to allow a split in node")  
            st.markdown("- **Criteria:** Will determine how splits are made in the tree. Entropy and the Gini Index are best for classification, whill log loss is recommended for binary probabilities")
            max_depth = st.slider("**Maximum Depth**", 1, 15, 5) #slider for selecting the max depth of DT
            min_samples_split = st.slider("**Minimum Samples to Split**", 1, 10, 5) #creating a slider so users can select the minimum numbers of samples they want for splits 
            crit = st.selectbox("**Criteria**", ["gini", "entropy", "log_loss"]) #which criteria will be used for evaluating 
            if crit == "gini": #using if else statement to write text explanation of each criteria so it will appear depending on which option is selected
                st.write("Measure of diversity based on the probability of certain values being in a node. A lower score, closer to 0, indicates a purer and better split. Lower gini index = better split.")
            elif crit == "entropy":
                st.write("Measures the diversity of the set, if the data split results in a lower combined entropy, it’s a better split. Lower average entropy = better split.")
            else:
                st.write("An evaluation metric for binary classification models. Lower log loss = predicted probabilities are more accurate in the model.")
#making the DT model and fitting
            dmodel = DecisionTreeClassifier(max_depth=max_depth, criterion=crit) 
            dmodel.fit(X_train, y_train) 
            y_pred = dmodel.predict(X_test)
#evaluating the model
            st.subheader("Metrics and Classification Report:")
            st.markdown("- **Accuracy:** Shows the percentage of correctly classified data at each split")
            st.markdown("- **Precision:** Looks at the predicted positives, showing the percentage of how many ***predicted positives*** were actually positive as a decimal")
            st.markdown("- **Recall:** The true positive rate, showing how many of all the ***real positives*** were correctly identified by the model")
            st.markdown("- **F1 Score:** Balances the accuracy between precision and recall, mirrors the precision and recall scores, if they are high, the F1-score will be high. Measured on a scale from 0-1, a higher F1 score is better, and indicates the model is able to better predict true values, and minimized the false values.")
            st.text(classification_report(y_test, y_pred))
#Visualizing the Tree
            st.subheader("Decision Tree Visualization")
#creating the actual decision tree visualization
            class_0_label = st.text_input("**What is class 0?**") #determining classes based on user input rather than writing them into the code
            class_1_label = st.text_input("**What is class 1?**")#determining classes based on user input rather than writing them into the code
            dot_data = tree.export_graphviz(dmodel, 
                feature_names= X.columns,
                class_names= [class_0_label, class_1_label], #here the class names selected by user are integrated into the DT visualization
                filled=True)
            graph = graphviz.Source(dot_data)
            st.graphviz_chart(dot_data) #showing the chart
#brief explanation of what the tree shows
            st.markdown("**The metrics shown include:**") #brief explanation of what visualization shows
            st.markdown("- **Features:** Which features are being used to split. *Note: Putting a constaint on the depth is helpful so the program starts finding more efficient ways of predicting, otherwise doesn't take into account the other nodes.*")
            st.markdown("- **Samples:** The number of observations included in each node")
            st.markdown("- **Criteria:** Gini, Entropy, or Log loss, depending on which criteria was previously selected")
            st.markdown("- **Class:** The class distributions within each node")
##ROC and AUC curves
            if len(np.unique(y)) == 2:#so only for binary
                st.subheader("**Below is the ROC Curve**")
                st.markdown("- The ROC, or Reciever Operating Characteristic, Curve shows a plot of the True Positive Rate and the False Positive Rate. This is helpful for selecting the threshold, where the number of true positives can be maximized, while also keeping a low percentage of false positives.")
                st.markdown("- The AUC, or area under the curve, is a summary of the model performance. The AUC score is on a scale from 0-1, and should be better than 0.5, which represents a 50 percent chance of correct prediction, meaning that the model is not better than random guessing.")
                y_probs = dmodel.predict_proba(X_test)[:, 1] #probabilities 
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
                
                st.write(f"**ROC Curve AUC Score: {roc_auc:.2f}**")
                #Plotting using mat lib
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                st.pyplot(plt)# Using st.pyplot() instead of show so works with streamlit

##Logistic regression
# Initializing and training model
        elif mlselect == "Logistic Regression":
            if len(np.unique(y)) != 2: #means that if the unique values are not equal to 2/ is not binary, print out this message because logistic regression will only work with binary target variables. Here, i am using np.unique(y) as a function to count the number of unique values 
                st.error("Logistic Regression only works with a binary target. Select a target column with only 2 unique values.") #creating an error message so users understand why it won't work if not binary
            else:
                st.markdown("You've selected Logistic regression.")
                lormodel = LogisticRegression()
                lormodel.fit(X_train, y_train)
                y_pred = lormodel.predict(X_test) #predict on test data


#metrics for evaluating the model: accuracy, classification report, confusion matrix, and roc and auc curves
                 
                st.subheader("Metrics and Classification Report:")
                st.text(classification_report(y_test, y_pred)) #creating classification report 
                st.markdown("- **Accuracy:** Shows the percentage of correctly classified data at each split")
                st.markdown("- **Precision:** Looks at the predicted positives, showing the percentage of how many ***predicted positives*** were actually positive as a decimal")
                st.markdown("- **Recall:** The true positive rate, showing how many of all the ***real positives*** were correctly identified by the model")
                st.markdown("- **F1 Score:** Balances the accuracy between precision and recall, mirrors the precision and recall scores, if they are high, the F1-score will be high. Measured on a scale from 0-1, a higher F1 score is better, and indicates the model is able to better predict true values, and minimized the false values.")
##confusion matrix creation
                st.write("### Confusion Matrix")
                st.markdown("Evaluates performance by comparing predictions from the model to actual values. The key parts of the confusion matrix are the **true negatives** (located in the top left corner), and the **true positives** (located in the bottom right corner). These show the number of times the model correctly predicts something that is actually negative or actually positive.")
                cm = confusion_matrix(y_test, y_pred) #creating the confusion matrix
                plt.figure(figsize=(6, 4)) ##then used matplotlib turn into a workable/ better looking confusion matrix
                sns.heatmap(cm, annot=True, cmap="Blues") #cmap to change the color scheme
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(plt) #showing the confusion matrix

                ##The Roc Curve 
                st.subheader("**Below is the ROC Curve**")
                st.markdown("- **The ROC,** or Reciever Operating Characteristic, Curve shows a plot of the True Positive Rate and the False Positive Rate. This is helpful for selecting the threshold, where the number of true positives can be maximized, while also keeping a low percentage of false positives.")
                st.markdown("- **The AUC,** The AUC, or area under the curve, summarizes the model performance. The AUC score is on a scale from 0-1, and should be better than 0.5, which represents a 50 percent chance of correct prediction. If the score is better than 0.5, it means that the model is better at predicting true positives than random guessing.")
                y_probs = lormodel.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs) #so only if binary
                st.write(f"**ROC Curve AUC Score: {roc_auc:.2f}**")
            #Plotting
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                st.pyplot(plt)
#Linear Regression
        elif mlselect == "Linear Regression":
            st.markdown("You've selected linear regression, view the metrics below.")
            #fitting and training the model
            linmodel = LinearRegression()
            linmodel.fit(X_train, y_train) 
            y_pred = linmodel.predict(X_test)
#finding the mse, rmse, and r squared
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            st.subheader("Linear Regression Metrics")
            st.markdown(f"**Mean Squared Error (MSE): {mse:.2f}**")
            st.markdown("- The MSE shows the average squared difference between actual and predicted values, where a lower value means a better model.")
            st.markdown(f"**Root Mean Squared Error (RMSE): {rmse:.2f}**")
            st.markdown("- The RMSE shows the difference between actual and predicted values, in the same units as the target, a lower value means a better model.")
            st.markdown(f"**R² Score: {r2:.2f}**")
            st.markdown("- The R² score shows the proportion of variance that the model accounts for. On a scale from 0 - 1, a score closer to 1 indictates a model that better accounts for variance.")


##Scaled vs unscaled? How to account for this? Confused in general? 





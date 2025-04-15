##Machine Learning Streamlit App

#Importing necessary Packages
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
from sklearn.preprocessing import StandardScaler

import graphviz
# Creating a title for app and adding an explanation 
st.title("Supervised Machine Learning App")
st.markdown("Welcome to this Machine Learning App! To get started, select a sample dataset or upload your own CSV on the left side panel. You will then be shown a preview of the data, and can begin making choices about what type of model you would like to use. Some metrics for evaluation are shown, as well as some brief explanations of what different metrics show. For a more in depth understanding, you can view the reading materials linked on the connected Github README file.")

##Data
#Uploading user dataset code
st.sidebar.header("Upload or Choose a Dataset") #adding it to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) #upload file as csv
sam_data = st.sidebar.selectbox("Or choose a sample dataset", ["None", "Iris", "Titanic", "Penguins"]) #sample dataset options and default as none

# Loading sample datasets and creating the options for datasets. First, if user uploads dataset, second is Iris dataset, third is Titanic dataset, then penguins
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file) #If own dataset uploaded load this
    df = pd.get_dummies(df, columns= df.select_dtypes(exclude=['number']).columns.tolist(), drop_first=True) #changing categorical columns into dummy, pd.get_dummies converts categorical variables into dummy variables so have numeric variables, use drop_first to avoid dummy trap. Using exclude=['number'] so only applies to non-numeric columns and makes them into dummies. 
elif sam_data == "Iris":
    df = sns.load_dataset("iris") #if Iris selected load this
    df = df[df["species"].isin(["setosa", "virginica"])] #changing the species to only have two options so it works with binary classification
    df = pd.get_dummies(df, columns= df.select_dtypes(exclude=['number']).columns.tolist(), drop_first=True) #same as with uploaded_file
    st.sidebar.markdown("**About Iris Dataset:** Contains information about Iris flowers, specifically their measurements.") #adding brief info about data
    st.sidebar.markdown("For more information click [here](https://www.kaggle.com/datasets/uciml/iris)") #adding hyperlink to more information about dataset
elif sam_data == "Titanic":
    df = sns.load_dataset("titanic").dropna(axis=1)  #if titanic selected load this 
    df = pd.get_dummies(df, columns= df.select_dtypes(exclude=['number']).columns.tolist(), drop_first=True)
    st.sidebar.markdown("**About Titanic Dataset:** Contains information about passengers on the Titanic, including their class, fare price, gender and whether they survived.") #including brief info about the data 
    st.sidebar.markdown("For more information click [here](https://www.kaggle.com/datasets/yasserh/titanic-dataset)")
elif sam_data == "Penguins":
    df = sns.load_dataset("penguins") 
    df = df[df["species"].isin(["Gentoo", "Chinstrap"])] #changing the species to only have two options so it works with binary classification
    df.dropna(inplace=True)
    df = pd.get_dummies(df, columns= df.select_dtypes(exclude=['number']).columns.tolist(), drop_first=True)
    st.sidebar.markdown("**About Penguins Dataset:** Contains information about Penguin species living on different islands.")
    st.sidebar.markdown("For more information click [here](https://github.com/allisonhorst/palmerpenguins)")

if df is not None:
    st.write("**Preview of Dataset**", df.head()) #show a preview of df selected 

    # Selecting columns #if only want to use some columns, selecting here
    sel_col = st.selectbox("**Select the target column**", df.columns) #allowing user to pick what they want as the target column
    features = st.multiselect("**Select the features you would like to use:**", options=[col for col in df.columns if col != sel_col]) #allowing users to select multiple feature columns, besides the sel_col / the target column previously selected 

    if features:
        X = df[features] #defining X 
        y = df[sel_col] #defining y

        #Splitting the data into train and test groups 
        split = st.slider("**Split the Data into Training and Testing**", 10, 90, 20) / 100 #creating slider so users can select train/test split, lowest value 10, highest value 90, default to on 20 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
        st.markdown(f"Number in Training Set: {len(X_train)}") #using len and f to print out the number of training samples so it is more clear to the user 
        st.markdown(f"Number in Testing Set: {len(X_test)}") #using len and f to print out the number of testing samples so it is more clear to the user than just the slider percentages. Also easier for the user to see how many in each 

        #Model options 
        #providing brief descriptions of the 3 models offered
        st.header("Selecting a Model")
        st.markdown("**Choose which machine learning model you would like to use. Below is a brief explanation of each model:**")
        st.markdown("- **Decision Tree:** In simple terms, a decision tree outlines options based on whether a binary variable is true or false. Each answer leads to a lower level where the process is repeated, until a conclusion can be reached, represented by a leaf node, a box with no branches. This means that no more splits can occur in the tree. **This decision tree model is for classification, and is therefore good if the target variable is categorical.**")
        st.markdown("- **Linear Regression:** Can be used for linear relationships where the target is something on a continuous numeric scale, or a categorical numerical scale in some cases. A linear regression model is helpful for evaluating a relationship between variables, for example, a one unit increase in y increases x by 15 points. **A good model choice is the target variable is numeric.**")
        st.markdown("- **Logistic Regression:** To evaluate the probability of the features influencing the target variable. The target variable should be categorical, while the features can be categorical or numeric. A logistic regression may be the best option if the goal is to see how each selected feature impacts the outcome's probability. **A good model selection if the intended target variable is binary.**")
        mlselect = st.selectbox("", ["Decision Tree", "Linear Regression", "Logistic Regression"]) #creating a selectbox so users can pick which model they want to use
##Decision tree
#throughout the creation of the models, if, else and else if "elif" statements are used to make the code conditional on other parts. This is helpful since there are three model options to pick from 
        if mlselect == "Decision Tree": #if decision tree selected, this code will be run
            st.markdown("You've selected a decision tree. Below, you can adjust 3 hyperparameters. This decision tree model should be used for classification, please ***ensure you selected a categorical target variable***") #text that will appear once decision tree is selected as the model. Explaining that this is a classification decision tree so users select a categorical variable
            st.subheader("Hyperparameters") #header for the hyperparameters 
            st.markdown("- **Maximum depth:** Changes how many levels the decision tree will have, with a smaller number meaning there will be fewer levels. ***Tip: It may be a good idea to set max depth to the number of features involved***") #explaining the hyperparameters using markdown text
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
            dmodel = DecisionTreeClassifier(max_depth=max_depth, criterion=crit) #decision tree model
            dmodel.fit(X_train, y_train) #fitting the model
            y_pred = dmodel.predict(X_test) #creating the y prediction
#evaluating the model
            st.subheader("Metrics and Classification Report:") 
            st.markdown("- **Accuracy:** Shows the percentage of correctly classified data") #explaining what each of the key metrics shown mean 
            st.markdown("- **Precision:** Looks at the predicted positives, showing the percentage of how many ***predicted positives*** were actually positive as a decimal")
            st.markdown("- **Recall:** The true positive rate, showing how many of all the ***real positives*** were correctly identified by the model")
            st.markdown("- **F1 Score:** Balances the accuracy between precision and recall, mirrors the precision and recall scores, if they are high, the F1-score will be high. Measured on a scale from 0-1, a higher F1 score is better, and indicates the model is able to better predict true values, and minimized the false values.")
            st.text(classification_report(y_test, y_pred)) #creating the classification report
#Visualizing the Tree
            st.subheader("Decision Tree Visualization") #title
#creating the actual decision tree visualization
            class_0_label = st.text_input("**What is class 0?**") #determining classes based on user input rather than writing them into the code
            class_1_label = st.text_input("**What is class 1?**")#determining classes based on user input rather than writing them into the code
            dot_data = tree.export_graphviz(dmodel, #creating the visualization
                feature_names= X.columns,
                class_names= [class_0_label, class_1_label], #here the class names selected by user are integrated into the DT visualization
                filled=True)
            graph = graphviz.Source(dot_data)
            st.graphviz_chart(dot_data) #showing the chart
#brief explanation of what the tree shows
            st.markdown("**The metrics shown include:**") #brief explanation of what visualization shows
            st.markdown("- **Features:** Which features are being used to split. *Note: Putting a constraint on the depth is helpful so the program starts finding more efficient ways of predicting, otherwise doesn't take into account the other nodes.*")
            st.markdown("- **Samples:** The number of observations included in each node")
            st.markdown("- **Criteria:** Gini, Entropy, or Log loss, depending on which criteria was previously selected")
            st.markdown("- **Class:** The class distributions within each node")
##ROC and AUC curves
            if len(np.unique(y)) == 2:#so only for binary
                st.subheader("**Below is the ROC Curve**") #title for section 
                st.markdown("- The ROC, or Reciever Operating Characteristic, Curve shows a plot of the True Positive Rate and the False Positive Rate. This is helpful for selecting the threshold, where the number of true positives can be maximized, while also keeping a low percentage of false positives.") #explaining ROC curve
                st.markdown("- The AUC, or area under the curve, is a summary of the model performance. The AUC score is on a scale from 0-1, and should be better than 0.5, which represents a 50 percent chance of correct prediction, meaning that the model is not better than random guessing.") #explaining AUC curve
                y_probs = dmodel.predict_proba(X_test)[:, 1] #probabilities 
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
                
                st.write(f"**ROC Curve AUC Score: {roc_auc:.2f}**") #writing the AUC score created 
                #Plotting using mat lib
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})') #label with the AUC score code showing written previously
                plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
                plt.xlabel("False Positive Rate") #label
                plt.ylabel("True Positive Rate") #label
                plt.title("ROC Curve") #title
                plt.legend(loc="lower right") 
                st.pyplot(plt)# Using st.pyplot() instead of show so works with streamlit

##Logistic regression
# Initializing and training model
        elif mlselect == "Logistic Regression":
            if len(np.unique(y)) != 2: #means that if the unique values are not equal to 2/ is not binary, print out this message because logistic regression will only work with binary target variables. Here, i am using np.unique(y) as a function to count the number of unique values 
                st.error("Logistic Regression only works with a binary target. Please select a **binary target variable**.") #creating an error message so users understand why it won't work if not binary
            else:
                st.markdown("You've selected Logistic regression, please ensure you selected a ***binary target column***") #what will appear when selecting logistic regression
                lormodel = LogisticRegression() #creating the logistic regression model
                lormodel.fit(X_train, y_train)
                y_pred = lormodel.predict(X_test) #predict on test data


#metrics for evaluating the model: accuracy, classification report, confusion matrix, and roc and auc curves
                 
                st.subheader("Metrics and Classification Report:")
                st.text(classification_report(y_test, y_pred)) #creating classification report 
                st.markdown("- **Accuracy:** Shows the percentage of correctly classified data")
                st.markdown("- **Precision:** Looks at the predicted positives, showing the percentage of how many ***predicted positives*** were actually positive as a decimal")
                st.markdown("- **Recall:** The true positive rate, showing how many of all the ***real positives*** were correctly identified by the model")
                st.markdown("- **F1 Score:** Balances the accuracy between precision and recall, mirrors the precision and recall scores, if they are high, the F1-score will be high. Measured on a scale from 0-1, a higher F1 score is better, and indicates the model is able to better predict true values, and minimized the false values.")
##confusion matrix creation
                st.write("### Confusion Matrix") #title for confusion matrix section 
                st.markdown("Evaluates performance by comparing predictions from the model to actual values. The key parts of the confusion matrix are the **true negatives** (located in the top left corner), and the **true positives** (located in the bottom right corner). These show the number of times the model correctly predicts something that is actually negative or actually positive. Higher numbers, in comparision to the other boxes, may indicate a better model performance. Comparing the number of true negatives and true positives can also help see which a model is better at predicting.") #explaining what a confusion matrix is 
                cm = confusion_matrix(y_test, y_pred) #creating the confusion matrix
                plt.figure(figsize=(6, 4)) ##then used matplotlib turn into a workable/ better looking confusion matrix
                sns.heatmap(cm, annot=True, cmap="Blues") #cmap to change the color scheme
                plt.title("Confusion Matrix") #title
                plt.xlabel("Predicted") #label for x axis
                plt.ylabel("Actual") #label for y axis
                st.pyplot(plt) #showing the confusion matrix

                ##The Roc Curve 
                st.subheader("**Below is the ROC Curve**") #header for this section 
                st.markdown("- **The ROC,** or Reciever Operating Characteristic, Curve shows a plot of the True Positive Rate and the False Positive Rate. This is helpful for selecting the threshold, where the number of true positives can be maximized, while also keeping a low percentage of false positives.")
                st.markdown("- **The AUC,** The AUC, or area under the curve, summarizes the model performance. The AUC score is on a scale from 0-1, and should be better than 0.5, which represents a 50 percent chance of correct prediction. If the score is better than 0.5, it means that the model is better at predicting true positives than random guessing.")
                y_probs = lormodel.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs) #so only if binary
                st.write(f"**ROC Curve AUC Score: {roc_auc:.2f}**") #writing the roc auc score
            #Plotting
            ##same as in decision tree
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                st.pyplot(plt)
#Linear Regression
        elif mlselect == "Linear Regression": #if linear regression selected 
            st.markdown("You've selected linear regression, please ensure you selected a ***target variable that is on a continuous numeric scale*** before choosing a scaling option, and viewing the metrics below.")
            st.subheader("Scaling")
            st.markdown("- Using unscaled data can make it difficult to compare features measured in different units. By scaling the data, it becomes easier to draw comparisons across the different features.")
            st.markdown("- If scaled, and making comparisions across features, a larger coefficient indicates a larger impact on the target variable.")
            st.markdown("- If the features are already in the same units, scaling may be unnecessary. It is important to consider what is necessary for your data, and your objectives.")
            app_scale = st.radio("**Select whether you would like to scale the model**", ["Unscaled", "Scaled"]) #giving users the option to pick between scaled and unscaled through radio prompt with a circle for each option
            if app_scale == "Scaled": #if scaled, apply scaler 
                scaler = StandardScaler() #using standardscaler 
                X_train = scaler.fit_transform(X_train) #applying scaler to X_Train
                X_test = scaler.transform(X_test) #applying scaler to X_test
            #fitting and training the model
            linmodel = LinearRegression()
            linmodel.fit(X_train, y_train) 
            y_pred = linmodel.predict(X_test)
#finding the mse, rmse, and r squared
            mse = mean_squared_error(y_test, y_pred) #calculating mse
            rmse = np.sqrt(mse) #calculating rmse
            r2 = r2_score(y_test, y_pred) #calcuating r squared score 

            st.subheader("Linear Regression Metrics") #title for metrics section
            st.markdown(f"**Mean Squared Error (MSE): {mse:.2f}**") 
            st.markdown("- The MSE shows the average squared difference between actual and predicted values, where a lower value generally means a better model.") #explaining mse
            st.markdown(f"**Root Mean Squared Error (RMSE): {rmse:.2f}**")
            st.markdown("- The RMSE shows the difference between actual and predicted values, in the same units as the target, a lower value generally means a better model.") #explaining rmse
            st.markdown(f"**R² Score: {r2:.2f}**")
            st.markdown("- The R² score shows the proportion of variance that the model accounts for. On a scale from 0 - 1, a score closer to 1 indicates a model that better accounts for variance.") #explaining r squared








# Penguin Streamlit App

## 📕 Project Overview: 
**This is a basic Streamlit app that reads data and has interactive widgets in Streamlit. In this introductory Streamlit app, you can look at penguins based on their species, island location, body mass, and flipper length by interacting with various widgets**
- **Project goal:** The goal of this project is to begin to understand how to code a Streamlit app for evaluating a dataset. It is also helpful for familiarization with what widgets do and look like.
- **Skills and Packages:** Python, Streamlit, Pandas, data exploration
## 📖 Instructions 


**1.** Import the Pandas and Streamlit libraries using the code below: 
````
import pandas as pd
import streamlit as st
````
Below are the versions of the libraries used in this app:
````
pandas==2.2.3
seaborn==0.13.2
````
**2.** Run Streamlit in your terminal and open the prompted link to view the app. For more detailed instructions on this setup, click [here](https://docs.kanaries.net/topics/Streamlit/streamlit-vscode)


**3.** Once these steps are completed, you will be able to run the code in your own VS code environment and view the created app through the Streamlit link opened in your browser.

**Note:** In order to use this Streamlit app, the Python file must be downloaded and Streamlit must be used to create personal url. 

## 📲 App Features
- **Button:** A simple button in Streamlit to click to emphasize excitement for using the app and practice displaying text.
  - The button is shown below:
<img width="358" alt="Screenshot 2025-05-08 at 9 53 25 PM" src="https://github.com/user-attachments/assets/03f4ebd5-a7da-4250-b065-827125165444" />

- **Species Selection:** Allows the user to select different species of penguins and islands of residence to study
  - The dropdown selection for the Gentoo species selection is shown below: 
<img width="868" alt="Screenshot 2025-05-08 at 9 53 51 PM" src="https://github.com/user-attachments/assets/2af9ed27-082c-4356-913f-921db9d70f33" />

- **Sliders:** Allows the user to subset the penguin species based on mass, flipper and bill length. The selected values are then used to create tables of the penguins that match the criteria, showing all the features in the dataset. 
  - The image below shows the body weight selection slider and the chart created of the penguins that match the criteria that are outputted.
  <img width="773" alt="Screenshot 2025-05-08 at 7 55 09 PM" src="https://github.com/user-attachments/assets/b4db7895-fedc-4a1e-a8a4-ddc0ecd999b3" />
## ℹ️ The Data 
- This app uses a Seaborn dataset about different penguin species, where users can look at features like weight, flipper and bill length, and location.
- To learn more about the dataset used, click [here](https://github.com/allisonhorst/palmerpenguins) 




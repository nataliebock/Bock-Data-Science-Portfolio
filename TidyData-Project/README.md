# Tidy Data Project 
## 🧑‍🏫Project Overview  
- The goal of this project was to clean an untidy dataset using the principles of tidy data and then begin to analyze the cleaned data.
- For this project, I tidied and used a dataset on the 2008 Summer Olympics. The source where the untidy data was adapted from is linked below under References. 

### 🧹Tidy Data  
- What does it mean to "tidy data": In Hadley Wickham's work entitled "Tidy Data," she writes that "structuring datasets to facilitate manipulation, visualisation and modelling" (20). By tidying, the dataset is prepared for use and analysis through coding.
- The principles include: Every observation is in its own row, every variable has its own column with only one variable per column, and that each observational unit forms its own table.
- Why these principles are important: As written by Hadley Wickham, "This framework makes it easy to tidy messy datasets because only a small set of tools are needed to deal with a wide range of un-tidy datasets." Therefore, while the principles may be challenging to grasp at first, understanding the principles makes tidying data with the appropriate tools much easier! 
- Why tidy data is important: Tidying data creates a universal structure that makes it easy for other people to use and analyze the same dataset. Tidy data is important in coding because many tools will assume the dataset is tidy, and therefore for accurate analysis to be created, the dataset much match the expectations of the coding tools.
  
## 📖 Instructions 
My script was written using Google Colaboratory, a simple platform for coding in Python that can be attached to your Google Drive. The instructions provided below describe how to run the code using this platform. 
**1.** Add the dataset into your Google Drive.
**2.** "Mount" your Drive in the Google Colab script: This is done by clicking the folder icon on the left hand side of the opened Google Colab project, and then clicking the folder with the Google Drive icon located below the title "Files."
**3.** Load the dataset: Copy the dataset's path and assign it to a name in your code. It should look like this, but with your specific path to the dataset:
````
  medal = '/content/drive/MyDrive/Intro to Data Science/olympics_08_medalists.csv'
  medal = pd.read_csv(medal);
````
**4.** After completing these steps, the data should be ready to use in your Google Colab copy of the code!
**5.** In this project, I used the pandas, seaborn, and numpy packages. You can load them by copying the code below:
````
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
````
**6.** Once these steps are followed, simply click "Runtime," and then "Run All" to run the code.

## 📕Dataset Description 
This dataset contains information about the 2008 Summer Olympics.
- After cleaning, the variables included are the names of the competitor, their gender, the event they competed in, and the type of medal they won – if they did win a medal.
- The data comes from a dataset by Giorgio Comai (OBCT/CCI) for EDJNet, linked below. In the original data, country is also included as a variable, as well as a medal count per person. In this case, these variables were not included, showing that the main objective here is whether or not someone won a medal, and what type of medal, rather than how many they have won over time.

## 📷 Visuals 
Below, I have included 2 visualizations made for this project, as well as a brief explanation and analysis of what is being depicted. To visualize the data, I divided the events into three categories: land events with no ball, land events with a ball, and water events. Below, are the visualizations created for the water events.   
### This visualziation shows the number and type of medals won in each water-based event in the 2008 Summer Olympics.
![Image 3-17-25 at 3 19 PM](https://github.com/user-attachments/assets/48a94d57-9fc6-4e52-b74c-97a6ca934bf8)
- On the y-axis, the included events are listed, and the x-axis shows the number of medals won. The colors show the type of medal won, intuitively represented by colors similar to the medals (Silver is represented by grey, gold is represnted by yellow, and bronze is represented by red).
- The bar chart helps show that not all medal winners are included in this data set. For example, the modern pentathlon is missing one bronze medalist. This is important to remember when drawing conclusions from this data set. 
- As shown, the most medals were given out for the rowing and swimming events, while the fewest medals were given for the triathlon and the modern pentathlon.
- **Why so many:** In 2008, there were 14 rowing events, all of which consisted of teams. There were also 34 swimming events, some of which are relays and therefore contain more than one competitor. This helps to explain why so many of each type of medal were given out in these events.
- **Why so few:** The triathlon and modern pentathlon both only have two events, one for men and one for women, explaining why so few medals are given out in these events.

## ℹ️ References 
[Data Source](https://edjnet.github.io/OlympicsGoNUTS/2008/)
[Tidy Data by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
[Data Cleaning Reference Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
[Information about the Summer 2008 Olympics ](https://www.olympics.com/en/olympic-games/beijing-2008/results)


3. Create a README for the Tidy Data Project Repository
Content Suggestions:

Instructions: Include step-by-step instructions on how to run the notebook, along with dependencies (e.g., pandas, matplotlib, etc.).
Dataset Description: Outline the source of your data and any pre-processing steps.
References: Provide links to the cheat sheet and tidy data paper for further reading.
Visual Examples: Consider adding screenshots of your visualizations or code snippets.


# Tidy Data Project 
## Project Overview: 
- The goal of this project was to clean an untidy dataset using the principles of tidy data and then begin to analyze the cleaned data.
- For this project, I tidied and used a dataset on the 2008 Summer Olympics. The source where the untidy data was adapted from is linked below under References. 

### Tidy Data: 
- What does it mean to "tidy data": In Hadley Wickham's work entitled "Tidy Data," she writes that "structuring datasets to facilitate manipulation, visualisation and modelling" (20). By tidying, the dataset is prepared for use and analysis through coding.
- The principles include: Every observation is in its own row, every variable has its own column with only one variable per column, and that each observational unit forms its own table.
- Why these principles are important: As written by Hadley Wickham, "This framework makes it easy to tidy messy datasets because only a small set of tools are needed to deal with a wide range of un-tidy datasets." Therefore, while the principles may be challenging to grasp at first, understanding the principles makes tidying data with the appropriate tools much easier! 
- Why tidy data is important: Tidying data creates a universal structure that makes it easy for other people to use and analyze the same dataset. Tidy data is important in coding because many tools will assume the dataset is tidy, and therefore for accurate analysis to be created, the dataset much match the expectations of the coding tools.
  
## Instructions: 
My script was written using Google Colaboratory, a simple platform for coding in Python that can be attached to your Google Drive. The instructions provided below describe how to run the code using this platform. 
1. Add the dataset into your Google Drive.
2. "Mount" your Drive in the Google Colab script: This is done by clicking the folder icon on the left hand side of the opened Google Colab project, and then clicking the folder with the Google Drive icon located below the title "Files."
3. Load the dataset: Copy the dataset's path and assign it to a name in your code. It should look like this:
   ```
function test() {
  medal = '/content/drive/MyDrive/Intro to Data Science/olympics_08_medalists.csv'
  medal = pd.read_csv(medal);
}
```
## Dataset Description: 

## References: 

## Visuals: 

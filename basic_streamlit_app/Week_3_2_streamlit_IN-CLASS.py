# Import the Streamlit library
import streamlit as st

#Navigate through terminal 
#ls: how you look at whats inside of folder
#cd : lets u choose something in the directory to go to/ lets you go to a folder?

# Display a simple text message

# Display a large title on the app
st.title("Hello, streamlit!")
st.write("This is my first sreamlit app") #smaller writing 
st.markdown("## This is my first streamlit app") #the ## will make writing larger
# ------------------------
# INTERACTIVE BUTTON
# ------------------------

# Create a button that users can click.
# If the button is clicked, the message changes.
if st.button("Click me!"):
    st.write("You clicked the button, nice work!")
else:
    st.write("Go ahead, click the button. I dare you.")

# ------------------------
# COLOR PICKER WIDGET
# ------------------------

# Creates an interactive color picker where users can choose a color.
# The selected color is stored in the variable 'color'.

# Display the chosen color value

# ------------------------
# ADDING DATA TO STREAMLIT
# ------------------------

# Import pandas for handling tabular data

# Display a section title

# Create a simple Pandas DataFrame with sample data


# Display a descriptive message

# Display the dataframe in an interactive table.
# Users can scroll and sort the data within the table.

# ------------------------
# INTERACTIVE DATA FILTERING
# ------------------------

# Create a dropdown (selectbox) for filtering the DataFrame by city.
# The user selects a city from the unique values in the "City" column.

# Create a filtered DataFrame that only includes rows matching the selected city.

# Display the filtered results with an appropriate heading.
  # Show the filtered table

# ------------------------
# NEXT STEPS & CHALLENGE
# ------------------------

# Play around with more Streamlit widgets or elements by checking the documentation:
# https://docs.streamlit.io/develop/api-reference
# Use the cheat sheet for quick reference:
# https://cheat-sheet.streamlit.app/

### Challenge:
# 1️⃣ Modify the dataframe (add new columns or different data).
# 2️⃣ Add an input box for users to type names and filter results.
# 3️⃣ Make a simple chart using st.bar_chart().
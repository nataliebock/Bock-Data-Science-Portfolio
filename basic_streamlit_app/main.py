import streamlit as st

import pandas as pd
#Instructions: streamlit run basic_streamlit_app/main.py 

penguins = pd.read_csv("Data/penguins.csv")
##Adding title and explaining the app 
st.title("Penguins App")
st.text("- This app uses a dataset with information about penguins.")
st.text("- You can look at penguins based on species, island, and other facts!")
st.text("- In order to view information, click on the drop down menus and sliders.")

#A simply button encourage excitement about penguins! 

if st.button("Click if you are excited about penguins!"):
    st.write("Yay you clicked let's learn!")
else:
    st.write("You don't want to learn about penguins :(")
# Now the interactive widgets inclduing drop down/ select box and sliders 
st.subheader("Now, let's look at some penguins!")
#Full table:
st.write("Here's the full table:")
st.dataframe(penguins)
#by species and then island: 
species = st.selectbox("Select a species", penguins["species"].unique()) 

                      
filtered_pen = penguins[penguins["species"] == species]

st.write(f"Penguins that are {species}:") 
st.dataframe(filtered_pen)


island = st.selectbox("Select an island", penguins["island"].unique()) 
filtered_pen2 = penguins[penguins["island"] == island]

st.write(f"Penguins that are on {island}:") 
st.dataframe(filtered_pen2)

#Sliders for mass, flipper and bill length 
mass = st.slider("Choose a body mass range in grams:",
                   min_value = penguins["body_mass_g"].min(),
                   max_value = penguins["body_mass_g"].max())


st.write(f"Penguin weights less than {mass}:")
st.dataframe(penguins[penguins['body_mass_g'] <= mass])


flipper = st.slider("Choose a flipper length range in mm:",
                   min_value = penguins["flipper_length_mm"].min(),
                   max_value = penguins["flipper_length_mm"].max())


st.write(f"Penguin flippers shorter than {flipper}:")
st.dataframe(penguins[penguins['flipper_length_mm'] <= flipper])

bill = st.slider("Choose a bill length range in mm:",

                   min_value = penguins["bill_length_mm"].min(),
                   max_value = penguins["bill_length_mm"].max())


st.write(f"Penguin bills shorter than {bill}:")
st.dataframe(penguins[penguins['bill_length_mm'] <= bill])



import streamlit as st

import pandas as pd
penguins = pd.read_csv("Data/penguins.csv")

st.title("Penguins App")
st.text("- This app uses a dataset with information about penguins.")
st.text("- You can look at penguins based on species, island, and other facts!")
st.text("- In order to view information, click on the drop down menus and sliders.")

if st.button("Click if you are excited about penguins!"):
    st.write("Yay you clicked let's learn!")
else:
    st.write("You don't want to learn about penguins :(")

st.subheader("Now, let's look at some penguins!")

st.write("Here's the full table:")
st.dataframe(penguins)

species = st.selectbox("Select a species", penguins["species"].unique()) 

                      
filtered_pen = penguins[penguins["species"] == species]

st.write(f"Penguins that are {species}:") 
st.dataframe(filtered_pen)


island = st.selectbox("Select an island", penguins["island"].unique()) 
filtered_pen2 = penguins[penguins["island"] == island]

st.write(f"Penguins that are on {island}:") 
st.dataframe(filtered_pen2)


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



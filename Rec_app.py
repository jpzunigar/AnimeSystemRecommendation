import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_searchbox import st_searchbox
from typing import List
from numpy import load



@st.cache(persist=True)

def get_anime_df(url):
    df=pd.read_csv(url)
    df=df.reset_index()
    return df

def get_anime_index(df):
    indices = pd.Series(df.index, index=df['Name']).drop_duplicates()
    return indices

def get_anime_name_list(df):
    name_list=list(df["Name"])
    return name_list
def get_cosine_matrix(url):
    dict_data = load(url)
    dict_data = dict_data['arr_0']
    return dict_data
def get_recommendatios(title,matrix,indices,df):
    # Get the index of the input anime
    idx = indices[title]

    # Get the similarity scores of all movies with that anime
    sim_scores = list(enumerate(matrix[idx]))

    # Sort the animes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar animes, sim_scores[0] would be the anime itself
    sim_scores = sim_scores[1:11]

    # Get the anime indices
    anime_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['Name'].iloc[anime_indices]

    # Set the background color of the page

st.set_page_config(page_title='ASR',
                    page_icon=':bar_chart:',
                    layout='wide')

st.markdown("<style>body {background-color: #ADD8E6;}</style>", unsafe_allow_html=True)

# Create the title
title = "Anime System Recommendation"

# Center the title and set font size
st.markdown("<h1 style='text-align: center; font-size: 50px; color: #FF5733;'>%s</h1>" % title, unsafe_allow_html=True)


st.write("")
st.write("")
st.write("")


df=get_anime_df('./Data/anime_1000.csv')

def search_function(search_term: str) -> List[str]:
    suggestions = []
    # Loop through the list of countries
    for name in get_anime_name_list(df):
        # Check if the search term appears in the country name (case-insensitive)
        if search_term.lower() in name.lower():
            suggestions.append(name)
    return suggestions

# pass search function to searchbox
selected_value = st_searchbox(
    search_function,
    key="name_searchbox",
)

st.write(get_anime_index(df))

if selected_value is None:
    st.write('Seleccione un anime')
else:
    rec=get_recommendatios(selected_value,get_cosine_matrix('./Data/matrix_3000.npz'),get_anime_index(df),df)
    st.write(rec)

st.write(get_cosine_matrix('./Data/matrix_3000.npz'))
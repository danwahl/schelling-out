import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_WEIGHTS = {
    'biking': 1.0,
    'climbing': 1.0,
    'housing': 1.0,
    'vegan': 1.0,
    'winter': 1.0,
    'summer': 0.0,
}


@st.cache_data
def load_data():
    cities = pd.read_csv('cities.csv', index_col=[0, 1])
    data = pd.read_csv('data.csv', index_col=[0, 1])
    return cities, data


cities, data = load_data()
chicago = data.loc[('Chicago', 'Illinois')]
weights = pd.Series(DEFAULT_WEIGHTS)

st.markdown('# Schelling Out 2024')

with st.expander('Weights'):
    for key in weights.index:
        weights[key] = st.slider(key, 0.0, 1.0, DEFAULT_WEIGHTS[key], 0.05)
weights['housing'] *= -1

scores = weights*data.apply(lambda x: (x/chicago).apply(np.log2), axis=1)
scores['lifestyle'] = scores[[
    'biking', 'climbing', 'vegan', 'winter', 'summer']].mean(axis=1)
scores['total'] = scores[weights.index].mean(axis=1)
scores.sort_values('total', ascending=False, inplace=True)
scores = scores.replace([np.inf, -np.inf], np.nan).dropna()

st.markdown('## Map')

cities['total'] = scores['total']
cities = cities.dropna(subset=['total'])

fig = px.scatter_geo(cities,
                     lat='latitude',
                     lon='longitude',
                     hover_name=cities.index.map(
                         lambda x: f"{x[0]}, {x[1]}").to_list(),
                     hover_data={'2023 estimate': True, 'total': True,
                                 'latitude': False, 'longitude': False},
                     color='total',
                     color_continuous_scale='RdBu',
                     color_continuous_midpoint=0,
                     labels={'color': 'Score'},
                     scope='usa',
                     )
st.plotly_chart(fig, use_container_width=True)

st.markdown('## Plot')

fig = px.scatter(scores,
                 x='housing',
                 y='lifestyle',
                 hover_name=scores.index.map(
                     lambda x: f"{x[0]}, {x[1]}").to_list(),
                 color='total',
                 color_continuous_scale='RdBu',
                 color_continuous_midpoint=0,
                 labels={'x': 'Housing', 'y': 'Lifestyle'},
                 )
st.plotly_chart(fig, use_container_width=True)

st.markdown('## Scores')

st.dataframe(scores, use_container_width=True)

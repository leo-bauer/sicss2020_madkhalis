import plotly.graph_objects as go

import pandas as pd

import plotly.io as pio
pio.renderers.default = "browser"

df = pd.read_csv("research project\\raw data\\libya 6 groups1.csv")

df['text'] = df['location'] + ', ' + df['actor1'] + ', ' + df['assoc_actor_1'] + ', ' + df['actor2'] + ', ' + df['year'].astype(str)

fig = go.Figure(data=go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        text=df['text'],
        mode='markers',
        marker_color=df['number'],
        ))
fig.update_layout(
        title = 'Events at Libya by Non Governmental Actors<br>(Hover for location, actors, and time of the event)',
        geo_scope='africa',
    )

fig.show()


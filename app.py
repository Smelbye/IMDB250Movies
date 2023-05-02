import dash
import numpy as np
from dash import dcc
from dash import html
import matplotlib
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from collections import defaultdict


# Read the dataset
df = pd.read_csv("IMDB Top 250 Movies.csv")

# Preprocessing the data
df['year'] = pd.to_datetime(df['year'], format='%Y')
df['decade'] = (df['year'].dt.year // 10) * 10
df['casts'] = df['casts'].apply(lambda x: x.split(','))
df['directors'] = df['directors'].apply(lambda x: x.split(','))
df['writers'] = df['writers'].apply(lambda x: x.split(','))
df['genre'] = df['genre'].apply(lambda x: x.split(','))
df = df.explode('genre')

# App 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Functions for visualizations
def get_top_n(df, column, n, exclude=None):
    counter = defaultdict(int)
    for row in df[column]:
        for item in row:
            if exclude and exclude in item.strip():
                continue
            counter[item.strip()] += 1

    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:n]

def get_average_ratings(df, column, n, exclude=None):
    avg_ratings = {}
    for item in df[column]:
        for name in item:
            name = name.strip()
            if exclude and exclude in name:
                continue
            avg_ratings[name] = df.loc[df[column].apply(lambda x: name in x), 'rating'].mean()

    return sorted(avg_ratings.items(), key=lambda x: x[1], reverse=True)[:n]

def convert_runtime_to_minutes(runtime_str):
    hours = 0
    minutes = 0
    time_parts = runtime_str.split()
    for part in time_parts:
        if 'h' in part:
            hours = int(part[:-1])
        elif 'm' in part:
            minutes = int(part[:-1])
    total_minutes = hours * 60 + minutes
    return total_minutes

df['run_time'] = df['run_time'].apply(convert_runtime_to_minutes)

def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '').replace('$', '')
        if value.isdigit():
            return float(value)
    return None

df['budget'] = df['budget'].apply(convert_to_numeric)
df['box_office'] = df['box_office'].apply(convert_to_numeric).fillna(0)

#Custom Color Chart
num_colors = 9
colormap = mcolors.LinearSegmentedColormap.from_list("", ["aquamarine", "dodgerblue"])
colors = [mcolors.to_hex(colormap(i)) for i in np.linspace(0, 1, num_colors)]


# Bar Chart
@app.callback(
    Output('bar-chart', 'figure'),
    Input('dropdown', 'value'))
def update_bar_chart(value):
    top_10 = get_average_ratings(df, value, 10, exclude='actor')
    labels = [item[0] for item in top_10]
    values = [item[1] for item in top_10]

    fig = px.bar(x=labels, y=values, text=values, color=values, color_continuous_scale=colors)
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(title=dict(
            text=f"<b>Top 10 {value.capitalize()[:-1]} by Average Rating</b>",
            font=dict(size=24, color="mediumblue"),
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(title="Name", titlefont=dict(size=18, family="Arial", color="black")),
        yaxis=dict(title="Average Rating", titlefont=dict(size=18, family="Arial", color="black"),range=[0, 10]),
        plot_bgcolor='rgba(230, 230, 250, 0.3)', 
        yaxis_gridcolor='lightgrey', margin=dict(t=90),
        coloraxis_showscale=False)
    
    top_person = top_10[0]
    second_person = top_10[1]
    
    fig.add_annotation(
        dict(
            text=f"The Top {value.capitalize()[:-1]} <b>{top_person[0]}</b> has an average rating of <b>{top_person[1]:.2f}</b>.<br>The 2nd highest rated {value.capitalize()[:-1]} <b>{second_person[0]}</b> has an average rating of <b>{second_person[1]:.2f}</b>.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.2,
            xanchor="center",
            yanchor="top",
            font=dict(color="black", size=14)
        )
    )
    return fig

# Pie Chart
@app.callback(
    Output('avg-rating-pie-chart', 'figure'),
    Input('dropdown', 'value'))
def update_avg_rating_pie_chart(value):
    all_genres_df = df.assign(genre=df['genre'].str.split(',')).explode('genre')
    all_genres_ratings = all_genres_df.groupby('genre')['rating'].mean().reset_index()
    all_genres_ratings = all_genres_ratings[all_genres_ratings['genre'] != 'Music'] 
    top_genres_ratings = all_genres_ratings.nlargest(6, 'rating')

    fig = px.pie(top_genres_ratings, names='genre', values='rating', hole=0)
    rounded_ratings = top_genres_ratings['rating'].round(2).astype(str)
    fig.update_traces(text=top_genres_ratings['genre'] + ": " + rounded_ratings + " avg", textinfo='label', hoverinfo='label+value', insidetextorientation='radial',
                      marker=dict(colors=colors), showlegend=True, legendgroup=True)
    
    top_genre = top_genres_ratings.iloc[0]
    
    fig.update_layout(
        title=dict(
            text=f"<b>Average Rating by Genre</b>",
            font=dict(color="mediumblue", size=24),
            x=0.5
        ),
        plot_bgcolor='honeydew', margin=dict(t=100)
    )
    
    fig.add_annotation(
        dict(
            text=f"The top genre by rating is <b>{top_genre['genre']}</b>, with an average rating of <b>{top_genre['rating']:.2f}</b>.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.15,
            xanchor="center",
            yanchor="top",
            font=dict(color="black", size=14)
        )
    )

    return fig

# Horizontal Bar Chart
@app.callback(
    Output('movie-runtime-distribution-bar-chart', 'figure'),
    Input('dropdown', 'value'))
def update_movie_runtime_distribution_bar_chart(value):
    run_time_genre = df.groupby('genre')['run_time'].mean().reset_index().sort_values('run_time', ascending=True).tail(9)
    fig = px.bar(run_time_genre, x='run_time', y='genre', orientation='h', text='run_time',
                 color='run_time', color_continuous_scale=colors)
    fig.update_traces(texttemplate='%{text:.2f}', textposition='inside', textfont=dict(color='black'))
    fig.update_layout(
        title=dict(
            text=f"<b>Average Movie Runtime by Genres</b>",
            font=dict(size=24, color="mediumblue"),
            x=0.5
        ),
        xaxis=dict(title="Average Runtime in Minutes", titlefont=dict(size=18, family="Arial", color="black"),
                   tickvals=list(range(25, 150, 25)), ticktext=[str(x) for x in range(25, 150, 25)], showticklabels=True),
        yaxis=dict(title="Genre", titlefont=dict(size=18, family="Arial", color="black")),
        plot_bgcolor='rgba(230, 230, 250, 0.3)',
        yaxis_gridcolor='lightgrey', margin=dict(t=90),
        coloraxis=dict(colorbar=dict(title=""))
    )
    
    # Hide the 0 on the x-axis
    fig.update_xaxes(showticklabels=True, range=[0, max(fig.data[0].x) + 25], tick0=25, dtick=25)
    
    longest_runtime_genre = run_time_genre.iloc[-1]['genre']
    longest_runtime = run_time_genre.iloc[-1]['run_time']

    fig.add_annotation(
        text=f"Longest runtime by genre is <b>{longest_runtime_genre}</b> with an <br>average runtime of <b>{longest_runtime:.2f} minutes</b>.",
        xanchor='left', yanchor='bottom',
        x=0.05, y=1,
        xref='paper', yref='paper',
        font=dict(size=16, family="Arial", color="black"),
        showarrow=False
    )
    
    return fig




# Scatter Plot
@app.callback(
    Output('runtime-vs-rating-scatter-plot', 'figure'),
    Input('dropdown', 'value'))
def update_runtime_vs_rating_scatter_plot(value):
    fig = px.scatter(data_frame=df,
                     x='run_time',
                     y='rating',
                     hover_data=['name', 'year', 'rating', 'genre', 'certificate', 'run_time', 'tagline', 'casts', 'directors', 'writers', 'decade'],
                     color='rating',
                     size='box_office',
                     title='Movies Runtime vs Rating',
                     labels={'run_time': 'Runtime (minutes)', 'rating': 'IMDB Rating'},
                     color_continuous_scale=colors)

    fig.update_layout(
        title=dict(
            text=f"<b>Runtime vs. Rating</b>",
            font=dict(size=24, color="mediumblue"),
            x=0.5
        ),
        xaxis=dict(title="Runtime in Minutes", titlefont=dict(size=18, family="Arial", color="black")),
        yaxis=dict(title="Rating", titlefont=dict(size=18, family="Arial", color="black")),
        plot_bgcolor='rgba(230, 230, 250, 0.3)',
        yaxis_gridcolor='lightgrey',
        margin=dict(t=80),
        coloraxis=dict(colorbar=dict(title=""))  
    )

    fig.add_annotation(
        text=f"Movies between <b>130</b> to <b>200</b> Minutes tends to get higher ratings.",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.55,
        y=1.1,
        xanchor="center",
        yanchor="top",
        font=dict(color="black", size=14)
    )

    return fig


# Line Chart
@app.callback(
    Output('avg-rating-by-decade-line-chart', 'figure'),
    Input('dropdown', 'value'))
def update_avg_rating_by_decade_line_chart(value):
    avg_rating_by_decade = df.groupby('decade')['rating'].mean().reset_index()
    highest_decade = avg_rating_by_decade.loc[avg_rating_by_decade['rating'].idxmax()]
    second_highest_decade = avg_rating_by_decade.nlargest(2, 'rating').iloc[1]

    fig = px.line(avg_rating_by_decade, x='decade', y='rating', text='rating', markers=True, line_shape='spline')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='top center', line=dict(color='slateblue'))
    fig.update_layout(
        title=dict(
            text=f"<b>Average Rating of Movies Produced per Decade</b>",
            font=dict(color="mediumblue", size=24),
            x=0.5
        ),
        xaxis=dict(title="Decade", titlefont=dict(size=18, family="Arial", color="black")),
        yaxis=dict(title="Average Rating", titlefont=dict(size=18, family="Arial", color="black")),
        annotations=[
            dict(
                text=f"The decade <b>{int(highest_decade['decade'])}</b> had an average rating of <b>{highest_decade['rating']:.2f}</b>, which is the highest ever.<br>The second highest average rating is <b>{second_highest_decade['rating']:.2f}</b> in <b>{int(second_highest_decade['decade'])}</b>.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.1,
                xanchor="center",
                yanchor="top",
                font=dict(color="black", size=14)
            )
        ],
        width=1000, height=500,
        margin=dict(t=80, b=30, l=100, r=100),
        plot_bgcolor='rgba(230, 230, 250, 0.3)'
    )
    fig.update_layout(
        margin=dict(l=100, r=100, t=80, b=30, autoexpand=True),
        autosize=True,
    )
    return fig

footer_style = {
    'position': 'absolute',
    'bottom': '0',
    'width': '100%',
    'height': '35px',
    'line-height': '35px',
    'text-align': 'right',
    'color': 'grey',
    'font-size': '14px',
}

# Layout
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("IMDB Top 250 Movies Dashboard", className="text-center mt-4", style={"color": "mediumblue"})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H5("A visual analysis of IMDB's top 250 movies", className="text-center mb-4"),
                html.Div([
                    html.P([html.Strong("This dashboard contains five interactive visualizations"), html.Br()]),
                    html.P("1. Line Chart comparison of Average Rating per Decade"),
                    html.P("2. Vertical Bar Chart comparing Top 10 Writers / Directors by selection"),
                    html.P("3. Pie Chart comparing Top 6 Genres by Average Rating"),
                    html.P("4. Horizontal Bar Chart comparing Top Genres by Runtime"),
                    html.P("5. Scatter Chart displaying the relation between Runtime & Budget"),
                ], className="text-center mb-4")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='avg-rating-by-decade-line-chart', className="mx-auto", style={"width": "83.333%", "align": "center"}),
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='dropdown',
                    options=[
                        {'label': 'Directors', 'value': 'directors'},
                        {'label': 'Writers', 'value': 'writers'}
                    ],
                    value='writers',
                    style={'width': '40%'}
                ),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='bar-chart'),
            ], width=6),
            dbc.Col([
                dcc.Graph(id='avg-rating-pie-chart'),
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='movie-runtime-distribution-bar-chart'),
            ], width=6),
            dbc.Col([
                dcc.Graph(id='runtime-vs-rating-scatter-plot'),
            ], width=6)
        ])
    ], fluid=True),
    html.Footer("Dataset Source: https://www.kaggle.com/datasets/rajugc/imdb-top-250-movies-dataset", style=footer_style)
], style={'position': 'relative', 'min-height': '100vh'})

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080)

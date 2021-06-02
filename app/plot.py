"""
Plotting functions.
"""

import numpy as np
import plotly.express as px
import streamlit as st
from plotly import graph_objects as go

from .cluster_stats import compute_dist_matrix
from .embed import compute_2d, compute_embeddings, load_embedder
from .timer import timer


SIZE_POWER = 0.8


def clusters_chart(df, phrase_col='cluster_center',
                   hover_col='cluster_center',
                   size_col='cluster_size'):
    """Plot chart with clusters."""

    if len(df) == 1:
        st.warning('It is not possible to make plot with only one cluster.')
        return

    with timer('load_embedder inside clusters_chart', disable=False):
        embedder = load_embedder()
    with timer('compute_embeddings inside clusters_chart', disable=False):
        embeddings = compute_embeddings(embedder, df[phrase_col].tolist())
    with timer('compute_2d inside clusters_chart', disable=False):
        embeddings_2d = compute_2d(embeddings)

    df = df.copy()
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    df = df[df.cluster_size > 1]
    fig = px.scatter(df, x='x', y='y', hover_name=hover_col,
                     size=np.power(df[size_col], SIZE_POWER),
                     hover_data=df.columns,
                     color_discrete_sequence=['#7aaaf7'],
                     width=800, height=600)
    fig.layout.update(showlegend=False)
    st.plotly_chart(fig)


def plot_distance_histogram(max_distance, mean_distance):
    """Plot histograms for max and mean distances between point inside cluster."""

    fig = px.histogram(max_distance[max_distance > 0.01], nbins=50,
                       width=500, height=300,
                       title='Maximum distance')
    st.plotly_chart(fig)

    fig = px.histogram(mean_distance, nbins=50,
                       width=500, height=300,
                       title='Mean distance')
    st.plotly_chart(fig)


def plot_size_histogram(cluster_size):
    """Plot histogram for cluster size."""

    fig = px.histogram(x=cluster_size, nbins=50,
                       width=500, height=300)
    st.plotly_chart(fig)


def plot_dist_matrix(embeddings, phrases, width=800, height=600):
    """Plot distance matrix between embeddings."""

    dist_matrix = compute_dist_matrix(embeddings)

    fig = go.Figure(data=go.Heatmap(z=dist_matrix, x=phrases,
                                    y=phrases, colorscale='Viridis_r'))

    fig.update_layout(autosize=False, width=width, height=height,
                      xaxis_showticklabels=False, yaxis_showticklabels=False)

    st.plotly_chart(fig)

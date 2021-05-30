"""
Main application.
"""

import pandas as pd
import streamlit as st

import app.SessionState as SessionState
from app.cluster import make_clustering
from app.data import load_data
from app.download import csv_download_link
from app.embed import make_embeddings


st.set_page_config(page_title='Short texts clustering', layout='wide')


def main():

    session_state = SessionState.get(phrases=None, clusters=None,
                                     embeddings=None, df_clusters=None)

    st.sidebar.write("""This is demo for clustering of short texts.""")
    st.sidebar.write("""
                     Computing text embeddings using Universal Sentence Encoder,
                     running clustering model, showing results in different ways.""")
    st.sidebar.header('Menu')
    mode = st.sidebar.radio('Choose page', options=['Load data',
                                                    'Clustering',
                                                    'Download clusters'])

    if mode == 'Load data':

        load_data(session_state)
        with st.beta_expander('Show data'):
            st.write(pd.Series(session_state.phrases, name='phrases'))

        if session_state.phrases is not None:
            make_embeddings(session_state)

    elif mode == 'Clustering':

        if session_state.phrases is None:
            st.write('No data. Load data for clustering.')
            return

        make_clustering(session_state)

    elif mode == 'Download clusters':

        if session_state.clusters is None:
            st.write('No clusters trained.')
        else:
            st.write('Download phrases and corresponding clusters as csv file.')
            df = pd.DataFrame({'phrase': session_state.phrases,
                               'cluster_id': session_state.clusters})
            csv_download_link(df, sidebar=False)


if __name__ == '__main__':

    main()
